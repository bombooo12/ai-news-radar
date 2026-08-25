#!/usr/bin/env python3
"""
个性化选题推荐脚本
读取每日清洗后的新闻数据 + 产能画像 + 反馈历史
调用 DeepSeek API 生成个性化选题推荐
"""

import json
import os
import sys
import re
import yaml
import time
import urllib.request
from datetime import datetime, timezone, timedelta

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')

# DeepSeek API
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions'
DEEPSEEK_MODEL = 'deepseek-chat'


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

    if best_score < 2:
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

        if track_id and track_score >= 2:
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
    """分析反馈历史，提取偏好模式"""
    if not feedback_data:
        return {'selected_tracks': {}, 'selected_formats': {}, 'rejected_patterns': [], 'source_weights': {}}

    history = feedback_data.get('feedback_history', [])
    selected_tracks = {}
    selected_formats = {}
    rejected_patterns = []
    source_weights = {}

    for entry in history[-50:]:
        if entry.get('action') == 'selected':
            track = entry.get('track', 'unknown')
            selected_tracks[track] = selected_tracks.get(track, 0) + 1

            fmt = entry.get('format', '')
            if fmt:
                selected_formats[fmt] = selected_formats.get(fmt, 0) + 1

            source = entry.get('source', '')
            if source:
                source_weights[source] = source_weights.get(source, 0) + 0.1
        elif entry.get('action') == 'rejected':
            reason = entry.get('reason', '')
            if reason and reason not in rejected_patterns:
                rejected_patterns.append(reason)

    return {
        'selected_tracks': selected_tracks,
        'selected_formats': selected_formats,
        'rejected_patterns': rejected_patterns,
        'source_weights': source_weights,
        'total_feedback': len(history)
    }


def build_prompt(track_groups, capacity_profile, feedback_analysis, track_config):
    """构建 LLM 提示词"""
    now = datetime.now(timezone(timedelta(hours=8)))

    # 按赛道分组取 top items
    track_summaries = []
    for track_id, items in track_groups.items():
        track_name = track_config.get('tracks', {}).get(track_id, {}).get('name', track_id)
        track_desc = track_config.get('tracks', {}).get(track_id, {}).get('description', '')
        top_items = items[:10]

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

    # 反馈分析
    fb = feedback_analysis
    feedback_text = ''
    if fb['total_feedback'] > 0:
        tracks_pref = ', '.join([f'{k}:{v}次' for k, v in fb['selected_tracks'].items()]) or '暂无'
        formats_pref = ', '.join([f'{k}:{v}次' for k, v in fb['selected_formats'].items()]) or '暂无'
        rejected = ', '.join(fb['rejected_patterns']) or '暂无'
        feedback_text = f'''
## 用户反馈历史分析（共{fb['total_feedback']}条反馈）
- 最常选用的赛道: {tracks_pref}
- 最常选用的内容形式: {formats_pref}
- 拒绝模式: {rejected}
'''

    prompt = f'''你是一位资深的自媒体内容策划专家，请基于以下今日信息流和创作者画像，生成个性化选题推荐。

## 今日日期
{now.strftime('%Y-%m-%d %H:%M')} (北京时间)

## 创作者产能画像
{capacity_profile}

{feedback_text}

## 今日信息流（按赛道分组）

{all_tracks}

## 任务要求

请从以上信息中挑选 5-8 个最适合该创作者的选题，输出 JSON 格式的推荐列表。

### 选题筛选标准
1. 必须匹配创作者的产能（能写图文/能做视频/能做测评教程/能写深度分析，不做真人出镜口播）
2. 优先选择该创作者历史反馈中偏好的赛道和形式
3. 避开创作者历史反馈中拒绝的模式
4. 考虑时效性、受众需求、传播潜力
5. 每个选题必须有明确的内容形式和创作角度

### 输出格式（严格 JSON）
{{
  "date": "{now.strftime('%Y-%m-%d')}",
  "recommendations": [
    {{
      "rank": 1,
      "title": "选题标题（吸引人的）",
      "track": "赛道ID",
      "track_name": "赛道名",
      "format": "图文|视频|测评/教程|深度分析",
      "angle": "创作角度，一句话说明从什么视角切入",
      "target_audience": "目标受众",
      "reason": "为什么推荐这个选题（50字内）",
      "source_items": ["相关新闻标题1", "相关新闻标题2"],
      "source_urls": ["url1", "url2"],
      "urgency": "high|medium|low",
      "viral_potential": "high|medium|low"
    }}
  ],
  "weekly_plan": {{
    "monday": "周一建议发布的选题标题",
    "wednesday": "周三建议发布的选题标题",
    "friday": "周五建议发布的选题标题",
    "strategy": "本周内容策略建议（一句话）"
  }},
  "insights": "今日信息流整体观察（一句话，指出趋势或机会）"
}}

只输出 JSON，不要输出其他内容。'''

    return prompt


def call_deepseek(prompt):
    """调用 DeepSeek API"""
    if not DEEPSEEK_API_KEY:
        print('Warning: DEEPSEEK_API_KEY not set, skipping LLM recommendation')
        return None

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
    }

    payload = {
        'model': DEEPSEEK_MODEL,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.7,
        'max_tokens': 4096,
        'response_format': {'type': 'json_object'}
    }

    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(DEEPSEEK_API_URL, data=data, headers=headers, method='POST')

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode('utf-8'))
            content = result['choices'][0]['message']['content']
            return json.loads(content)
    except Exception as e:
        print(f'Error calling DeepSeek API: {e}')
        return None


def generate_fallback_recommendations(track_groups, track_config):
    """API 不可用时的降级推荐（基于评分排序）"""
    now = datetime.now(timezone(timedelta(hours=8)))
    recommendations = []

    for track_id, items in track_groups.items():
        track_name = track_config.get('tracks', {}).get(track_id, {}).get('name', track_id)
        for item in items[:2]:
            recommendations.append({
                'rank': len(recommendations) + 1,
                'title': item.get('title', 'N/A'),
                'track': track_id,
                'track_name': track_name,
                'format': '图文',
                'angle': '待AI分析',
                'target_audience': 'AI初学者',
                'reason': f'相关度{item.get("track_score", 0)}分',
                'source_items': [item.get('title', '')],
                'source_urls': [item.get('url', '')],
                'urgency': 'medium',
                'viral_potential': 'medium'
            })

    return {
        'date': now.strftime('%Y-%m-%d'),
        'recommendations': recommendations[:8],
        'weekly_plan': {
            'strategy': '降级模式：基于评分排序，未使用AI分析'
        },
        'insights': 'API不可用，展示评分排序结果',
        'fallback': True
    }


def main():
    print('=== 个性化选题推荐 ===')

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
    print(f'Scored {len(processed)} items across tracks')

    # 3. 按赛道分组
    track_groups = {}
    for item in processed:
        track = item['track']
        if track not in track_groups:
            track_groups[track] = []
        track_groups[track].append(item)

    for track_id, items in track_groups.items():
        track_name = track_config['tracks'][track_id]['name']
        print(f'  {track_name}: {len(items)} items')

    # 4. 分析反馈
    feedback_analysis = analyze_feedback(feedback_data)
    if feedback_analysis['total_feedback'] > 0:
        print(f'Feedback: {feedback_analysis["total_feedback"]} entries analyzed')

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

    output_path = os.path.join(DATA_DIR, 'recommendations.json')
    save_json(output_path, recommendations)
    print(f'Saved recommendations to {output_path}')

    # 7. 同时保存评分后的数据（带赛道标签）
    scored_path = os.path.join(DATA_DIR, 'latest-24h-scored.json')
    save_json(scored_path, processed)
    print(f'Saved scored data to {scored_path}')

    print('=== Done ===')


if __name__ == '__main__':
    main()
