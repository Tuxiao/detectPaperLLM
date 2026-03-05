package com.detectanyllm.client.model;

import com.alibaba.fastjson2.annotation.JSONField;
import java.util.List;

/**
 * 预测请求 DTO — 对应 API POST /predict 的 JSON body.
 */
public class PredictRequest {

    /** 单条文本（与 texts 二选一，也可同时指定） */
    private String text;

    /** 批量文本列表 */
    private List<String> texts;

    /** 决策模式覆盖：threshold | pm */
    @JSONField(name = "decision_mode")
    private String decisionMode;

    /** 阈值覆盖 */
    private Double threshold;

    /** 最大 token 长度覆盖 */
    @JSONField(name = "max_length")
    private Integer maxLength;

    /** 扰动采样数覆盖 */
    @JSONField(name = "num_perturb_samples")
    private Integer numPerturbSamples;

    /** KNN 邻居数覆盖 */
    @JSONField(name = "k_neighbors")
    private Integer kNeighbors;

    // ---- constructors ----

    public PredictRequest() {}

    /** 快捷构造：单文本 */
    public PredictRequest(String text) {
        this.text = text;
    }

    /** 快捷构造：批量文本 */
    public PredictRequest(List<String> texts) {
        this.texts = texts;
    }

    // ---- getters & setters ----

    public String getText() { return text; }
    public void setText(String text) { this.text = text; }

    public List<String> getTexts() { return texts; }
    public void setTexts(List<String> texts) { this.texts = texts; }

    public String getDecisionMode() { return decisionMode; }
    public void setDecisionMode(String decisionMode) { this.decisionMode = decisionMode; }

    public Double getThreshold() { return threshold; }
    public void setThreshold(Double threshold) { this.threshold = threshold; }

    public Integer getMaxLength() { return maxLength; }
    public void setMaxLength(Integer maxLength) { this.maxLength = maxLength; }

    public Integer getNumPerturbSamples() { return numPerturbSamples; }
    public void setNumPerturbSamples(Integer numPerturbSamples) { this.numPerturbSamples = numPerturbSamples; }

    public Integer getKNeighbors() { return kNeighbors; }
    public void setKNeighbors(Integer kNeighbors) { this.kNeighbors = kNeighbors; }
}
