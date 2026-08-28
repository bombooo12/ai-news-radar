#!/usr/bin/env python3
"""
个性化选题推荐脚本 v2（优化版）
改进：
1. 赛道阈值从2提到3，减少噪音
2. 每赛道取top15而非top10，给LLM更多选择
3. temperature从0.7降到0.5，推荐更稳定
4. 加入赛道平衡约束（每赛道至少1条推荐）
5. 来源链接扩展到5个
6. 反馈分析更精细（加入拒绝原因模式匹配）
7. 生成 ai_adjustments 字段，让用户看到AI的微调逻辑
"""

import json
import os
import sys
import re
import yaml
import time
import urllib.request
from datetime import datetime, timezone, timedelta
from collections import Counter

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')

# 主链路：硅基流动（DeepSeek-V3.2，TTFT 稳定）
PRIMARY_API_KEY = os.environ.get('SILICONFLOW_API_KEY', '') or os.environ.get('DEEPSEEK_API_KEY', '')
PRIMARY_API_URL = 'https://api.siliconflow.cn/v1/chat/completions'
PRIMARY_MODEL = 'deepseek-ai/DeepSeek-V3.2'

# 备用链路：默认复用 DEEPSEEK_* 仓库变量（改指百炼/腾讯云等平台即可切换）
BACKUP_API_KEY = os.environ.get('BACKUP_API_KEY', '') or os.environ.get('DEEPSEEK_API_KEY', '')
BACKUP_API_BASE = os.environ.get('BACKUP_API_BASE_URL', '') or os.environ.get('DEEPSEEK_API_BASE_URL', '') or 'https://api.siliconflow.cn/v1'
BACKUP_API_URL = BACKUP_API_BASE.rstrip('/') + '/chat/completions'
BACKUP_MODEL = os.environ.get('BACKUP_MODEL', '') or os.environ.get('DEEPSEEK_MODEL', '') or PRIMARY_MODEL

# 优化参数
TRACK_SCORE_THRESHOLD = 3      # 赛道评分阈值（从2提到3）
TOP_ITEMS_PER_TRACK = 10       # 每赛道取top条目（流式传输已解决超时）
LLM_TEMPERATURE = 0.5          # LLM温度（从0.7降到0.5）
MAX_SOURCE_LINKS = 5          # 来源链接数（从2扩展到5）
MAX_FEEDBACK_HISTORY = 100     # 保留最近100条反馈
FIRST_TOKEN_TIMEOUT = 90       # 等待首token超时即断开重试（应对硅基高峰排队）
LLM_CALL_TIMEOUT = 600         # 单次流式调用整体超时（秒），超时后降级为本地fallback推荐


def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'Warning: Could not load {filepath}: {e}')
        return None


def load_text(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f'Warning: Could not load {filepath}: {e}')
        return ''


def load_yaml(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f'Warning: Could not load {filepath}: {e}')
        return None


def save_json(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def score_item_by_track(item, track_config):
    """对单条新闻按各赛道关键词评分"""
    if not isinstance(item, dict):
        return {}
    text = ' '.join([
        str(item.get('title', '')),
        str(item.get('title_zh', '')),
        str(item.get('title_bilingual', '')),
        str(item.get('source', '')),
        str(item.get('site_name', '')),
        str(item.get('ai_label', '')),
    ]).lower()
    scores = {}

    for track_id, track in track_config.get('tracks', {}).items():
        total = 0
        matched = []
        for kw in track.get('keywords', []):
            word = kw['word'].lower()
            if word in text:
                total += kw['weight']
                matched.append(word)
        balance = track_config.get('track_balance', {}).get(track_id, 1.0)
        scores[track_id] = {
            'score': total * balance,
            'matched_keywords': matched,
            'track_name': track.get('name', track_id)
        }

    return scores


def assign_track(scores):
    """分配主赛道（得分最高的赛道）"""
    best_track = None
    best_score = 0
    for track_id, info in scores.items():
        if info['score'] > best_score:
            best_score = info['score']
            best_track = track_id

    if best_score < TRACK_SCORE_THRESHOLD:
        return None, 0, []

    track_info = scores[best_track]
    return best_track, track_info['score'], track_info['matched_keywords']


def process_news_items(items, track_config):
    """处理所有新闻条目，分配赛道"""
    processed = []
    for item in items:
        if not isinstance(item, dict):
            continue
        scores = score_item_by_track(item, track_config)
        track_id, track_score, matched_kw = assign_track(scores)

        if track_id and track_score >= TRACK_SCORE_THRESHOLD:
            processed.append({
                **item,
                'track': track_id,
                'track_name': scores[track_id]['track_name'],
                'track_score': track_score,
                'matched_keywords': matched_kw,
                'all_track_scores': {k: v['score'] for k, v in scores.items() if v['score'] > 0}
            })

    processed.sort(key=lambda x: x.get('track_score', 0), reverse=True)
    return processed


def analyze_feedback(feedback_data):
    """分析反馈历史，提取偏好模式（精细版）"""
    if not feedback_data:
        return {
            'selected_tracks': {},
            'selected_formats': {},
            'rejected_patterns': [],
            'rejected_titles': [],
            'source_weights': {},
            'total_feedback': 0,
            'selection_rate': 0,
            'notes': []
        }

    history = feedback_data.get('feedback_history', [])
    selected_tracks = Counter()
    selected_formats = Counter()
    rejected_patterns = []
    rejected_titles = []
    source_weights = {}
    notes = []

    selected_count = 0
    rejected_count = 0

    for entry in history[-MAX_FEEDBACK_HISTORY:]:
        action = entry.get('action', '')
        if action == 'selected':
            selected_count += 1
            track = entry.get('track', 'unknown')
            selected_tracks[track] += 1

            fmt = entry.get('format', '')
            if fmt:
                selected_formats[fmt] += 1

            source = entry.get('source', '')
            if source:
                source_weights[source] = source_weights.get(source, 0) + 0.1

            note = entry.get('note', '')
            if note:
                notes.append(note)

        elif action == 'rejected':
            rejected_count += 1
            reason = entry.get('reason', '') or entry.get('note', '')
            if reason and reason not in rejected_patterns:
                rejected_patterns.append(reason)
            rejected_titles.append(entry.get('topic_title', ''))

        elif action == 'maybe':
            note = entry.get('note', '')
            if note:
                notes.append(note)

    total = selected_count + rejected_count
    selection_rate = (selected_count / total * 100) if total > 0 else 0

    return {
        'selected_tracks': dict(selected_tracks),
        'selected_formats': dict(selected_formats),
        'rejected_patterns': rejected_patterns,
        'rejected_titles': rejected_titles[-10:],
        'source_weights': source_weights,
        'total_feedback': len(history),
        'selection_rate': round(selection_rate, 1),
        'notes': notes[-20:]
    }


def build_prompt(track_groups, capacity_profile, feedback_analysis, track_config):
    """构建 LLM 提示词（优化版）"""
    now = datetime.now(timezone(timedelta(hours=8)))

    # 按赛道分组取 top items（从10提到15）
    track_summaries = []
    for track_id, items in track_groups.items():
        track_name = track_config.get('tracks', {}).get(track_id, {}).get('name', track_id)
        track_desc = track_config.get('tracks', {}).get(track_id, {}).get('description', '')
        top_items = items[:TOP_ITEMS_PER_TRACK]

        items_text = []
        for item in top_items:
            title = item.get('title', 'N/A')
            source = item.get('source', 'N/A')
            url = item.get('url', '')
            score = item.get('track_score', 0)
            matched = ', '.join(item.get('matched_keywords', []))
            items_text.append(f'  - [{title}] 来源:{source} 相关度:{score} 关键词:{matched} 链接:{url}')

        track_summaries.append(f'### {track_name}（{track_desc}）\n' + '\n'.join(items_text))

    all_tracks = '\n\n'.join(track_summaries)

    # 反馈分析（精细版）
    fb = feedback_analysis
    feedback_text = ''
    if fb['total_feedback'] > 0:
        tracks_pref = ', '.join([f'{k}:{v}次' for k, v in sorted(fb['selected_tracks'].items(), key=lambda x: -x[1])]) or '暂无'
        formats_pref = ', '.join([f'{k}:{v}次' for k, v in sorted(fb['selected_formats'].items(), key=lambda x: -x[1])]) or '暂无'
        rejected = '; '.join(fb['rejected_patterns'][:5]) or '暂无'
        notes_str = '; '.join(fb['notes'][:5]) or '暂无'
        feedback_text = f'''
## 用户反馈历史分析（共{fb['total_feedback']}条反馈，选用率{fb['selection_rate']}%）
- 最常选用的赛道: {tracks_pref}
- 最常选用的内容形式: {formats_pref}
- 拒绝模式: {rejected}
- 用户点评摘录: {notes_str}
- 最近拒绝的选题: {'; '.join(fb['rejected_titles'][-3:]) if fb['rejected_titles'] else '暂无'}
'''

    # 赛道列表
    track_list = ', '.join([f'{tid}({t.get("name","")})' for tid, t in track_config.get('tracks', {}).items()])

    prompt = f'''你是一位资深的自媒体内容策划专家，擅长从不同角度拆解同一个选题。请基于以下今日信息流和创作者画像，生成个性化选题推荐，每个选题必须提供三个创作角度。

## 今日日期
{now.strftime('%Y-%m-%d %H:%M')} (北京时间)

## 创作者产能画像
{capacity_profile}

{feedback_text}

## 今日信息流（按赛道分组）

{all_tracks}

## 任务要求

请从以上信息中挑选 5-8 个最适合该创作者的选题，每个选题提供三个不同的创作角度（实操教程、避坑指南、深度解读）。

### 选题筛选标准
1. 必须匹配创作者的产能（能写图文/能做视频/能做测评教程/能写深度分析，不做真人出镜口播）
2. 优先选择该创作者历史反馈中偏好的赛道和形式
3. 避开创作者历史反馈中拒绝的模式和类似选题
4. 考虑时效性、受众需求、传播潜力
5. 每个赛道尽量至少推荐1条，保证赛道平衡
6. 每个选题必须有三个不同角度的创作方案
7. 每个选题的 source_urls 提供3-5个相关链接

### 三个角度的要求
- **实操教程**：面向初学者，教怎么用/怎么操作，步骤清晰，有具体方法
- **避坑指南**：指出常见误区、陷阱、风险，提供规避方法
- **深度解读**：分析背后的逻辑、趋势、影响，提供有深度的思考

### 输出格式（严格 JSON，不要 markdown 代码块）
{{
  "date": "{now.strftime('%Y-%m-%d')}",
  "recommendations": [
    {{
      "rank": 1,
      "title": "选题主标题（吸引人的总标题）",
      "track": "赛道ID",
      "track_name": "赛道名",
      "target_audience": "目标受众",
      "reason": "为什么推荐这个选题（50字内）",
      "source_items": ["相关新闻标题1", "相关新闻标题2", "相关新闻标题3"],
      "source_urls": ["url1", "url2", "url3", "url4", "url5"],
      "urgency": "high|medium|low",
      "viral_potential": "high|medium|low",
      "angles": [
        {{
          "name": "实操教程",
          "icon": "tutorial",
          "format": "图文|视频|测评",
          "hook": "这个角度的吸引人标题",
          "summary": "一句话说清教什么",
          "key_points": ["核心步骤1", "核心步骤2", "核心步骤3"],
          "difficulty": "入门",
          "estimated_time": "30分钟"
        }},
        {{
          "name": "避坑指南",
          "icon": "warning",
          "format": "图文|短视频",
          "hook": "这个角度的吸引人标题",
          "summary": "一句话说清避什么坑",
          "pitfalls": ["坑点1说明", "坑点2说明", "坑点3说明"],
          "difficulty": "入门",
          "estimated_time": "15分钟"
        }},
        {{
          "name": "深度解读",
          "icon": "deep",
          "format": "图文|视频",
          "hook": "这个角度的吸引人标题",
          "summary": "一句话说清深度讲什么",
          "insights": ["洞察1", "洞察2", "洞察3"],
          "difficulty": "中级",
          "estimated_time": "1小时"
        }}
      ]
    }}
  ],
  "weekly_plan": {{
    "monday": "周一建议发布的选题标题",
    "wednesday": "周三建议发布的选题标题",
    "friday": "周五建议发布的选题标题",
    "strategy": "本周内容策略建议（一句话）"
  }},
  "insights": "今日信息流整体观察（一句话，指出趋势或机会）",
  "ai_adjustment": "基于用户反馈的自主微调说明（一句话，说明本次推荐相比上次做了什么调整，如果没有反馈历史则说'首次运行，暂无微调'）"
}}

只输出 JSON，不要输出其他内容，不要用 ```json ``` 包裹。'''

    return prompt


def _stream_once(prompt, api_key, api_url, model, label, disable_thinking=False):
    """单次流式调用，成功返回解析后的 JSON，失败/超时返回 None"""
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    payload = {
        'model': model,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'temperature': LLM_TEMPERATURE,
        'max_tokens': 8192,
        'response_format': {'type': 'json_object'},
        'stream': True
    }
    # 备用链路（百炼等）默认走思考模式，禁用后更快更干净
    if disable_thinking:
        payload['thinking'] = {'type': 'disabled'}

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(api_url, data=data, headers=headers, method='POST')

    try:
        # timeout 同时约束连接与每次读块：首token迟迟不来（服务端排队）会触发 socket 超时
        resp = urllib.request.urlopen(req, timeout=FIRST_TOKEN_TIMEOUT)
        content = ''
        deadline = time.time() + LLM_CALL_TIMEOUT
        for line in resp:
            if time.time() > deadline:
                print(f'Error: {label} streaming exceeded {LLM_CALL_TIMEOUT}s')
                resp.close()
                return None
            line = line.decode('utf-8').strip()
            if not line:
                continue
            if line.startswith('data: '):
                chunk = line[6:]
                if chunk == '[DONE]':
                    break
                try:
                    chunk_data = json.loads(chunk)
                    delta = chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    if delta:
                        content += delta
                except json.JSONDecodeError:
                    continue
        resp.close()

        if not content:
            print(f'Error: {label} empty response')
            return None
        return json.loads(content)
    except Exception as e:
        print(f'Error: {label} call failed: {e}')
        return None


def call_deepseek(prompt):
    """主链路优先，首token超时/报错后切备用链路，都失败返回 None 走降级推荐"""
    if not PRIMARY_API_KEY and not BACKUP_API_KEY:
        print('Warning: No API key set (SILICONFLOW_API_KEY / DEEPSEEK_API_KEY), skipping LLM recommendation')
        return None

    attempts = [
        (PRIMARY_API_KEY, PRIMARY_API_URL, PRIMARY_MODEL, 'primary'),
        (BACKUP_API_KEY, BACKUP_API_URL, BACKUP_MODEL, 'backup'),
    ]
    for api_key, api_url, model, label in attempts:
        if not api_key:
            print(f'Warning: {label} API key not set, skipping')
            continue
        print(f'Calling LLM ({label}: {model})')
        result = _stream_once(prompt, api_key, api_url, model, label,
                              disable_thinking=(label == 'backup'))
        if result:
            print(f'  {label} LLM OK')
            return result
        print(f'  {label} failed' + ('，切换备用链路' if label == 'primary' else '，使用降级推荐'))
    return None


def generate_fallback_recommendations(track_groups, track_config):
    """API 不可用时的降级推荐"""
    now = datetime.now(timezone(timedelta(hours=8)))
    recommendations = []

    for track_id, items in track_groups.items():
        track_name = track_config.get('tracks', {}).get(track_id, {}).get('name', track_id)
        for item in items[:2]:
            title = item.get('title', 'N/A')
            source_urls = [item.get('url', '')]
            for extra in items[:MAX_SOURCE_LINKS]:
                u = extra.get('url', '')
                if u and u not in source_urls:
                    source_urls.append(u)

            recommendations.append({
                'rank': len(recommendations) + 1,
                'title': title,
                'track': track_id,
                'track_name': track_name,
                'target_audience': 'AI初学者/创作者',
                'reason': f'赛道相关度{item.get("track_score", 0)}分',
                'source_items': [i.get('title', '') for i in items[:MAX_SOURCE_LINKS]],
                'source_urls': source_urls[:MAX_SOURCE_LINKS],
                'urgency': 'medium',
                'viral_potential': 'medium',
                'angles': [
                    {
                        'name': '实操教程',
                        'icon': 'tutorial',
                        'format': '图文',
                        'hook': f'{title} 入门教程',
                        'summary': '从零开始上手的完整步骤',
                        'key_points': ['了解基本概念', '掌握核心操作', '动手实践案例'],
                        'difficulty': '入门',
                        'estimated_time': '30分钟'
                    },
                    {
                        'name': '避坑指南',
                        'icon': 'warning',
                        'format': '图文',
                        'hook': f'{title} 常见坑点',
                        'summary': '新手最容易踩的几个坑',
                        'pitfalls': ['概念理解误区', '操作步骤错漏', '预期管理不当'],
                        'difficulty': '入门',
                        'estimated_time': '15分钟'
                    },
                    {
                        'name': '深度解读',
                        'icon': 'deep',
                        'format': '图文',
                        'hook': f'{title} 深度分析',
                        'summary': '背后的逻辑和影响',
                        'insights': ['技术原理简析', '行业影响分析', '未来趋势展望'],
                        'difficulty': '中级',
                        'estimated_time': '1小时'
                    }
                ]
            })

    return {
        'date': now.strftime('%Y-%m-%d'),
        'recommendations': recommendations[:8],
        'weekly_plan': {
            'strategy': '降级模式：基于评分排序，AI多角度分析暂不可用'
        },
        'insights': 'API不可用，展示评分排序结果，多角度分析为模板生成',
        'ai_adjustment': '降级模式，暂无AI微调',
        'fallback': True
    }


def main():
    print('=== 个性化选题推荐 v2（优化版） ===')

    # 1. 加载数据
    raw_data = load_json(os.path.join(DATA_DIR, 'latest-24h-all.json'))
    if not raw_data:
        print('Error: No news data found')
        sys.exit(1)
    if isinstance(raw_data, list):
        news_data = raw_data
    elif isinstance(raw_data, dict):
        if 'items_all' in raw_data:
            news_data = raw_data['items_all']
        elif 'items' in raw_data:
            news_data = raw_data['items']
        else:
            news_data = []
    else:
        news_data = []
    print(f'Loaded {len(news_data)} news items')

    capacity_profile = load_text(os.path.join(CONFIG_DIR, 'capacity_profile.md'))
    track_config = load_yaml(os.path.join(CONFIG_DIR, 'track_keywords.yaml'))
    feedback_data = load_json(os.path.join(DATA_DIR, 'feedback.json'))

    if not track_config:
        print('Error: No track config found')
        sys.exit(1)

    # 2. 按赛道评分
    processed = process_news_items(news_data, track_config)
    print(f'Scored {len(processed)} items across tracks (threshold: {TRACK_SCORE_THRESHOLD})')

    # 3. 按赛道分组
    track_groups = {}
    for item in processed:
        track = item['track']
        if track not in track_groups:
            track_groups[track] = []
        track_groups[track].append(item)

    for track_id, items in track_groups.items():
        track_name = track_config['tracks'][track_id]['name']
        print(f'  {track_name}: {len(items)} items (top {TOP_ITEMS_PER_TRACK} sent to LLM)')

    # 4. 分析反馈
    feedback_analysis = analyze_feedback(feedback_data)
    if feedback_analysis['total_feedback'] > 0:
        print(f'Feedback: {feedback_analysis["total_feedback"]} entries analyzed')
        print(f'  Selection rate: {feedback_analysis["selection_rate"]}%')
        print(f'  Preferred tracks: {feedback_analysis["selected_tracks"]}')
        print(f'  Rejected patterns: {feedback_analysis["rejected_patterns"][:3]}')

    # 5. 调用 LLM 生成推荐
    prompt = build_prompt(track_groups, capacity_profile, feedback_analysis, track_config)
    recommendations = call_deepseek(prompt)

    if not recommendations:
        print('Using fallback recommendations')
        recommendations = generate_fallback_recommendations(track_groups, track_config)

    # 6. 保存
    recommendations['generated_at'] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    recommendations['total_items'] = len(news_data)
    recommendations['scored_items'] = len(processed)
    recommendations['tracks_summary'] = {
        track_id: len(items) for track_id, items in track_groups.items()
    }
    recommendations['engine_version'] = 'v2'
    recommendations['engine_params'] = {
        'track_score_threshold': TRACK_SCORE_THRESHOLD,
        'top_items_per_track': TOP_ITEMS_PER_TRACK,
        'llm_temperature': LLM_TEMPERATURE,
        'max_source_links': MAX_SOURCE_LINKS
    }

    output_path = os.path.join(DATA_DIR, 'recommendations.json')
    save_json(output_path, recommendations)
    print(f'Saved recommendations to {output_path}')
    print(f'  - {len(recommendations.get("recommendations", []))} topics with 3 angles each')
    if 'ai_adjustment' in recommendations:
        print(f'  - AI adjustment: {recommendations["ai_adjustment"]}')

    # 7. 同时保存评分后的数据
    scored_path = os.path.join(DATA_DIR, 'latest-24h-scored.json')
    save_json(scored_path, processed)
    print(f'Saved scored data to {scored_path}')

    print('=== Done ===')


if __name__ == '__main__':
    main()
