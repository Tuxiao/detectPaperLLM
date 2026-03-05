package com.detectanyllm.client.judge;

import com.detectanyllm.client.model.PredictResponse;
import com.detectanyllm.client.model.PredictionResult;

import java.util.List;
import java.util.Objects;

/**
 * AI 文本检测判断器 — 业务逻辑核心.
 *
 * <p>
 * 负责将底层 API 返回的 DC 评分转换为分级风险判断结果。
 * 阈值来源于训练阶段在 dev 集上通过 MCC 目标函数选定的最优值。
 *
 * <h3>阈值说明</h3>
 * <ul>
 * <li><b>最优阈值 (0.2012)</b> — dev 集 MCC 最优，precision ≈ 91%，recall ≈ 79%</li>
 * <li><b>高精度阈值 (1.3203)</b> — dev 集 precision ≈ 95%，recall ≈ 56%</li>
 * </ul>
 *
 * <h3>使用示例</h3>
 * 
 * <pre>{@code
 * DetectionJudge judge = DetectionJudge.withDefaultThresholds();
 * PredictResponse resp = client.predict(texts);
 * List<DetectionVerdict> verdicts = judge.judge(resp);
 * verdicts.forEach(v -> System.out.println(v.riskLevel() + " " + v.dc()));
 * }</pre>
 */
public class DetectionJudge {

    /**
     * 最优阈值：根据 16扰动/256截断 的 API 重新校准，F1 最优时的 DC 阈值.
     * <p>
     * 旧版：0.201171875；新版：1.1821
     * DC > 此值判定为 MACHINE_SUSPECT（疑似 AI 生成）
     * </p>
     */
    public static final double DEFAULT_OPTIMAL_THRESHOLD = 1.1821;

    /**
     * 高精度阈值：dev 集 (旧版) precision ≥ 95% 时的 DC 阈值.
     * <p>
     * DC > 此值判定为 MACHINE_HIGH_RISK（高风险 AI 生成）
     * </p>
     */
    public static final double DEFAULT_HIGH_PRECISION_THRESHOLD = 1.3203125;

    private final double optimalThreshold;
    private final double highPrecisionThreshold;

    /**
     * 使用自定义阈值构造判断器.
     *
     * @param optimalThreshold       最优阈值（DC > 此值 → 疑似 AI）
     * @param highPrecisionThreshold 高精度阈值（DC > 此值 → 高风险 AI）
     * @throws IllegalArgumentException 如果 highPrecisionThreshold ≤ optimalThreshold
     */
    public DetectionJudge(double optimalThreshold, double highPrecisionThreshold) {
        if (highPrecisionThreshold <= optimalThreshold) {
            throw new IllegalArgumentException(
                    "highPrecisionThreshold (" + highPrecisionThreshold
                            + ") must be greater than optimalThreshold (" + optimalThreshold + ")");
        }
        this.optimalThreshold = optimalThreshold;
        this.highPrecisionThreshold = highPrecisionThreshold;
    }

    /**
     * 使用训练阶段确定的默认阈值创建判断器.
     */
    public static DetectionJudge withDefaultThresholds() {
        return new DetectionJudge(DEFAULT_OPTIMAL_THRESHOLD, DEFAULT_HIGH_PRECISION_THRESHOLD);
    }

    // -----------------------------------------------------------------------
    // 判断方法
    // -----------------------------------------------------------------------

    /**
     * 对单条预测结果进行风险判断.
     *
     * @param result API 返回的单条预测结果
     * @return 包含风险等级的判断结果
     */
    public DetectionVerdict judge(PredictionResult result) {
        Objects.requireNonNull(result, "result must not be null");

        double dc = result.getDc();
        RiskLevel level;
        double hitThreshold;

        if (dc > highPrecisionThreshold) {
            level = RiskLevel.MACHINE_HIGH_RISK;
            hitThreshold = highPrecisionThreshold;
        } else if (dc > optimalThreshold) {
            level = RiskLevel.MACHINE_SUSPECT;
            hitThreshold = optimalThreshold;
        } else {
            level = RiskLevel.HUMAN;
            hitThreshold = optimalThreshold;
        }

        return new DetectionVerdict(result.getText(), dc, level, hitThreshold);
    }

    /**
     * 对批量预测响应进行风险判断.
     *
     * @param response API 返回的批量预测响应
     * @return 判断结果列表，与输入顺序一一对应
     */
    public List<DetectionVerdict> judge(PredictResponse response) {
        Objects.requireNonNull(response, "response must not be null");
        return response.getPredictions().stream()
                .map(this::judge)
                .toList();
    }

    // -----------------------------------------------------------------------
    // Getters
    // -----------------------------------------------------------------------

    /** 当前使用的最优阈值 */
    public double getOptimalThreshold() {
        return optimalThreshold;
    }

    /** 当前使用的高精度阈值 */
    public double getHighPrecisionThreshold() {
        return highPrecisionThreshold;
    }

    @Override
    public String toString() {
        return String.format("DetectionJudge{optimal=%.6f, highPrecision=%.6f}",
                optimalThreshold, highPrecisionThreshold);
    }
}
