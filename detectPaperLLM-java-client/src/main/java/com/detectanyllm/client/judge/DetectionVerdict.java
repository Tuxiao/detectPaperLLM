package com.detectanyllm.client.judge;

/**
 * 单条文本的检测判定结果.
 *
 * <p>
 * 将底层 API 返回的 DC 评分与业务层阈值结合，给出最终的风险等级判定。
 *
 * @param text         原始文本
 * @param dc           差异系数 (Discrepancy Coefficient)
 * @param riskLevel    风险等级
 * @param hitThreshold 命中的阈值（DC 超过此值触发当前等级判定），HUMAN 时为最优阈值
 */
public record DetectionVerdict(
        String text,
        double dc,
        RiskLevel riskLevel,
        double hitThreshold) {

    /** 是否判定为 AI 生成（含疑似和高风险） */
    public boolean isMachineGenerated() {
        return riskLevel.isMachineGenerated();
    }

    @Override
    public String toString() {
        return String.format("%s | DC=%.4f | \"%.40s...\"",
                riskLevel, dc, text);
    }
}
