"""
质量评分器

两种评分模式:
1. 规则评分 (Rule-based): 基于 checkpoint 关键词匹配, 快速但粗略
2. Judge 评分 (LLM-as-Judge): 使用强模型评判, 准确但需要额外 API 调用

最终得分 = 0.4 * 规则得分 + 0.6 * Judge 得分 (如果启用 Judge)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .tasks.coding_plans import Task, Checkpoint


@dataclass
class CheckpointScore:
    """单个检查点的评分"""
    checkpoint: Checkpoint
    passed: bool
    score: float  # 0.0 - 1.0
    matched_keywords: list[str] = field(default_factory=list)
    details: str = ""


@dataclass
class QualityScore:
    """完整的质量评分"""
    task_id: str
    rule_score: float = 0.0  # 规则评分 (0-100)
    judge_score: float = 0.0  # Judge 评分 (0-100)
    final_score: float = 0.0  # 最终得分 (0-100)
    checkpoint_scores: list[CheckpointScore] = field(default_factory=list)
    judge_feedback: str = ""
    structure_score: float = 0.0  # 结构性评分
    completeness_score: float = 0.0  # 完整性评分


class RuleScorer:
    """基于规则的评分器"""

    def score(self, task: Task, response: str) -> QualityScore:
        """对模型输出进行规则评分"""
        result = QualityScore(task_id=task.id)
        response_lower = response.lower()

        total_weight = 0.0
        earned_weight = 0.0

        for cp in task.checkpoints:
            cp_score = self._score_checkpoint(cp, response, response_lower)
            result.checkpoint_scores.append(cp_score)
            total_weight += cp.weight
            earned_weight += cp_score.score * cp.weight

        # 规则得分
        if total_weight > 0:
            result.rule_score = (earned_weight / total_weight) * 100

        # 结构性评分
        result.structure_score = self._score_structure(response)

        # 完整性评分
        result.completeness_score = self._score_completeness(response)

        # 最终得分 (仅规则模式)
        result.final_score = (
            result.rule_score * 0.6
            + result.structure_score * 0.2
            + result.completeness_score * 0.2
        )

        return result

    def _score_checkpoint(self, cp: Checkpoint, text: str, text_lower: str) -> CheckpointScore:
        """评估单个检查点"""
        matched = []
        for kw in cp.keywords:
            # 大小写不敏感匹配
            if kw.lower() in text_lower:
                matched.append(kw)
            # 也尝试正则匹配 (处理带括号等特殊字符)
            elif re.search(re.escape(kw), text, re.IGNORECASE):
                matched.append(kw)

        # 至少匹配一个关键词就算部分通过
        match_ratio = len(matched) / max(1, len(cp.keywords))

        if cp.is_critical:
            # 关键检查点: 至少匹配 50% 关键词才算通过
            passed = match_ratio >= 0.5
            score = match_ratio if passed else match_ratio * 0.3
        else:
            passed = match_ratio >= 0.3
            score = match_ratio

        return CheckpointScore(
            checkpoint=cp,
            passed=passed,
            score=score,
            matched_keywords=matched,
            details=f"匹配 {len(matched)}/{len(cp.keywords)} 关键词",
        )

    def _score_structure(self, text: str) -> float:
        """评估输出的结构性 (0-100)"""
        score = 0.0

        # 检查标题/分节
        headers = len(re.findall(r'^#{1,4}\s+', text, re.MULTILINE))
        if headers >= 3:
            score += 25
        elif headers >= 1:
            score += 15

        # 检查列表
        lists = len(re.findall(r'^[\s]*[-*\d.]+\s+', text, re.MULTILINE))
        if lists >= 5:
            score += 25
        elif lists >= 2:
            score += 15

        # 检查代码块
        code_blocks = len(re.findall(r'```', text))
        if code_blocks >= 4:
            score += 25
        elif code_blocks >= 2:
            score += 15

        # 检查总长度 (太短说明不够详细)
        length = len(text)
        if length >= 2000:
            score += 25
        elif length >= 1000:
            score += 15
        elif length >= 500:
            score += 10

        return min(100, score)

    def _score_completeness(self, text: str) -> float:
        """评估完整性 (0-100) - 适配代码生成/修复/审查等多种任务"""
        score = 0.0
        tl = text.lower()

        # 有实际代码输出
        if any(kw in tl for kw in ["def ", "func ", "class ", "function ", "const ", "import ", "from "]):
            score += 25

        # 有解释/分析
        if any(kw in tl for kw in ["because", "reason", "因为", "原因", "bug", "fix", "issue",
                                     "problem", "solution", "approach", "方案", "分析"]):
            score += 25

        # 有边界/错误考虑
        if any(kw in tl for kw in ["edge", "error", "exception", "invalid", "none", "null",
                                     "empty", "boundary", "边界", "异常", "错误"]):
            score += 25

        # 有测试/验证或完整输出
        if any(kw in tl for kw in ["test", "assert", "example", "verify", "验证", "测试",
                                     "return", "output", "result"]):
            score += 25

        return min(100, score)


class JudgeScorer:
    """基于 LLM 的评分器 (LLM-as-Judge)"""

    JUDGE_PROMPT = """你是一位资深软件架构师, 请评估以下 coding plan 的质量。

## 任务
{task_title}

## 任务描述
{task_prompt}

## 参考方案
{reference_plan}

## 被评估的方案
{response}

## 评估标准
请从以下维度评分 (每项 0-20 分, 总分 100):

1. **正确性** (0-20): 技术方案是否正确, 有无明显错误
2. **完整性** (0-20): 是否覆盖了所有需求要点
3. **深度** (0-20): 技术细节是否充分, 是否有真正的工程洞察
4. **可执行性** (0-20): plan 是否具体到可以直接编码实现
5. **边界思考** (0-20): 是否考虑了边界情况、错误处理、性能等

## 输出格式
请严格按以下 JSON 格式输出, 不要输出其他内容:
```json
{{
  "correctness": <0-20>,
  "completeness": <0-20>,
  "depth": <0-20>,
  "executability": <0-20>,
  "edge_thinking": <0-20>,
  "total": <0-100>,
  "feedback": "<简要评语, 50字以内>"
}}
```"""

    def build_judge_messages(self, task: Task, response: str) -> list[dict]:
        """构建 Judge 评分的消息"""
        prompt = self.JUDGE_PROMPT.format(
            task_title=task.title,
            task_prompt=task.prompt,
            reference_plan=task.reference_plan or "无参考方案",
            response=response,
        )
        return [
            {"role": "system", "content": "你是一位公正的评审专家。请严格按照评估标准打分。"},
            {"role": "user", "content": prompt},
        ]

    def parse_judge_response(self, judge_output: str) -> tuple[float, str]:
        """解析 Judge 的评分输出"""
        import json

        # 尝试提取 JSON
        json_match = re.search(r'\{[^{}]*\}', judge_output, re.DOTALL)
        if not json_match:
            return 0.0, "无法解析 Judge 输出"

        try:
            data = json.loads(json_match.group())
            total = float(data.get("total", 0))
            feedback = data.get("feedback", "")
            return min(100, max(0, total)), feedback
        except (json.JSONDecodeError, ValueError):
            return 0.0, "JSON 解析失败"


def compute_final_score(
    rule_score: QualityScore,
    judge_score: Optional[float] = None,
    judge_feedback: str = "",
) -> QualityScore:
    """计算最终得分 (结合规则和 Judge)"""
    if judge_score is not None:
        rule_score.judge_score = judge_score
        rule_score.judge_feedback = judge_feedback
        rule_score.final_score = (
            rule_score.rule_score * 0.4 + judge_score * 0.6
        )
    return rule_score
