package com.detectanyllm.client.model;

import com.alibaba.fastjson2.annotation.JSONField;
import java.util.List;

/**
 * 预测响应 DTO — 对应 API POST /predict 的返回 JSON.
 */
public class PredictResponse {

    /** 所有文本的预测结果列表 */
    private List<PredictionResult> predictions;

    /** 服务端耗时（毫秒） */
    @JSONField(name = "elapsed_ms")
    private double elapsedMs;

    /** 决策模式：threshold | pm */
    @JSONField(name = "decision_mode")
    private String decisionMode;

    // ---- getters & setters ----

    public List<PredictionResult> getPredictions() {
        return predictions;
    }

    public void setPredictions(List<PredictionResult> predictions) {
        this.predictions = predictions;
    }

    public double getElapsedMs() {
        return elapsedMs;
    }

    public void setElapsedMs(double elapsedMs) {
        this.elapsedMs = elapsedMs;
    }

    public String getDecisionMode() {
        return decisionMode;
    }

    public void setDecisionMode(String decisionMode) {
        this.decisionMode = decisionMode;
    }

    @Override
    public String toString() {
        return String.format("PredictResponse{predictions=%d, elapsed=%.2fms, mode=%s}",
                predictions != null ? predictions.size() : 0, elapsedMs, decisionMode);
    }
}
