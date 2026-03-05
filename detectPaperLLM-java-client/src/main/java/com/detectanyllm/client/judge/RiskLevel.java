package com.detectanyllm.client.judge;

/**
 * AI 文本检测风险等级.
 *
 * <p>
 * 基于差异系数 (DC) 与训练阶段确定的阈值进行分级判断：
 * <ul>
 * <li>{@link #HUMAN} — DC ≤ 最优阈值，判定为人类撰写</li>
 * <li>{@link #MACHINE_SUSPECT} — DC 超过最优阈值但未达高精度阈值，疑似 AI 生成</li>
 * <li>{@link #MACHINE_HIGH_RISK} — DC 超过高精度阈值，高置信度 AI 生成</li>
 * </ul>
 */
public enum RiskLevel {

    /** 人类撰写 */
    HUMAN("人类撰写", "✅"),

    /** 疑似 AI 生成（精确率 ≈ 91%） */
    MACHINE_SUSPECT("疑似AI生成", "⚠️"),

    /** 高置信度 AI 生成（精确率 ≈ 95%） */
    MACHINE_HIGH_RISK("高风险AI生成", "🚨");

    private final String label;
    private final String icon;

    RiskLevel(String label, String icon) {
        this.label = label;
        this.icon = icon;
    }

    /** 中文标签 */
    public String getLabel() {
        return label;
    }

    /** 图标符号 */
    public String getIcon() {
        return icon;
    }

    /** 是否判定为 AI 生成（含疑似和高风险） */
    public boolean isMachineGenerated() {
        return this != HUMAN;
    }

    @Override
    public String toString() {
        return icon + " " + label;
    }
}
