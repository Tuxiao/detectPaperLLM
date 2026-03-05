package com.detectanyllm.client.judge;

import com.detectanyllm.client.model.PredictResponse;
import com.detectanyllm.client.model.PredictionResult;
import org.junit.jupiter.api.*;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DetectionJudge 单元测试.
 */
class DetectionJudgeTest {

    private DetectionJudge judge;

    @BeforeEach
    void setUp() {
        judge = DetectionJudge.withDefaultThresholds();
    }

    // ---- 辅助方法 ----

    private PredictionResult mockResult(String text, double dc) {
        PredictionResult r = new PredictionResult();
        r.setText(text);
        r.setDc(dc);
        r.setLabelPred("IGNORED"); // 服务端标签在方案 B 中被忽略
        return r;
    }

    // ---- 阈值判断测试 ----

    @Test
    @DisplayName("DC ≤ 最优阈值 → HUMAN")
    void testHuman() {
        DetectionVerdict v = judge.judge(mockResult("人类文本", 0.10));
        assertEquals(RiskLevel.HUMAN, v.riskLevel());
        assertFalse(v.isMachineGenerated());
    }

    @Test
    @DisplayName("DC 恰好等于最优阈值 → HUMAN（边界）")
    void testHumanBoundary() {
        DetectionVerdict v = judge.judge(mockResult("边界文本",
                DetectionJudge.DEFAULT_OPTIMAL_THRESHOLD));
        assertEquals(RiskLevel.HUMAN, v.riskLevel());
    }

    @Test
    @DisplayName("DC > 最优阈值 且 ≤ 高精度阈值 → MACHINE_SUSPECT")
    void testMachineSuspect() {
        DetectionVerdict v = judge.judge(mockResult("疑似AI文本", 0.50));
        assertEquals(RiskLevel.MACHINE_SUSPECT, v.riskLevel());
        assertTrue(v.isMachineGenerated());
    }

    @Test
    @DisplayName("DC 恰好等于高精度阈值 → MACHINE_SUSPECT（边界）")
    void testMachineSuspectBoundary() {
        DetectionVerdict v = judge.judge(mockResult("边界文本",
                DetectionJudge.DEFAULT_HIGH_PRECISION_THRESHOLD));
        assertEquals(RiskLevel.MACHINE_SUSPECT, v.riskLevel());
    }

    @Test
    @DisplayName("DC > 高精度阈值 → MACHINE_HIGH_RISK")
    void testMachineHighRisk() {
        DetectionVerdict v = judge.judge(mockResult("高风险AI文本", 2.50));
        assertEquals(RiskLevel.MACHINE_HIGH_RISK, v.riskLevel());
        assertTrue(v.isMachineGenerated());
    }

    @Test
    @DisplayName("DC 为负值 → HUMAN")
    void testNegativeDc() {
        DetectionVerdict v = judge.judge(mockResult("负值文本", -1.0));
        assertEquals(RiskLevel.HUMAN, v.riskLevel());
    }

    // ---- 批量判断测试 ----

    @Test
    @DisplayName("批量判断 — 结果数量与输入一致")
    void testBatchJudge() {
        PredictResponse response = new PredictResponse();
        response.setPredictions(List.of(
                mockResult("文本A", 0.05), // HUMAN
                mockResult("文本B", 0.80), // MACHINE_SUSPECT
                mockResult("文本C", 2.00) // MACHINE_HIGH_RISK
        ));
        response.setElapsedMs(100.0);
        response.setDecisionMode("threshold");

        List<DetectionVerdict> verdicts = judge.judge(response);

        assertEquals(3, verdicts.size());
        assertEquals(RiskLevel.HUMAN, verdicts.get(0).riskLevel());
        assertEquals(RiskLevel.MACHINE_SUSPECT, verdicts.get(1).riskLevel());
        assertEquals(RiskLevel.MACHINE_HIGH_RISK, verdicts.get(2).riskLevel());
    }

    // ---- 自定义阈值测试 ----

    @Test
    @DisplayName("自定义阈值 — 控制判断行为")
    void testCustomThresholds() {
        DetectionJudge custom = new DetectionJudge(1.0, 5.0);

        assertEquals(RiskLevel.HUMAN, custom.judge(mockResult("", 0.5)).riskLevel());
        assertEquals(RiskLevel.MACHINE_SUSPECT, custom.judge(mockResult("", 3.0)).riskLevel());
        assertEquals(RiskLevel.MACHINE_HIGH_RISK, custom.judge(mockResult("", 6.0)).riskLevel());
    }

    @Test
    @DisplayName("非法阈值 — 高精度阈值必须大于最优阈值")
    void testInvalidThresholds() {
        assertThrows(IllegalArgumentException.class, () -> new DetectionJudge(5.0, 3.0));
        assertThrows(IllegalArgumentException.class, () -> new DetectionJudge(5.0, 5.0));
    }

    // ---- Verdict 属性测试 ----

    @Test
    @DisplayName("DetectionVerdict 保留原始文本和 DC 值")
    void testVerdictProperties() {
        DetectionVerdict v = judge.judge(mockResult("保留文本内容测试", 0.30));
        assertEquals("保留文本内容测试", v.text());
        assertEquals(0.30, v.dc(), 0.001);
    }
}
