package com.detectanyllm.client.model;

import com.alibaba.fastjson2.annotation.JSONField;

/**
 * 单条文本的预测结果.
 */
public class PredictionResult {

    /** 原始文本 */
    private String text;

    /** DC 分数（Discrepancy Coefficient） */
    @JSONField(name = "d_c")
    private double dc;

    /** 概率估计 p(machine)，仅 pm 模式下有值 */
    @JSONField(name = "p_m")
    private Double pm;

    /** p(machine) - p(human) 差值 */
    private Double delta;

    /** 近邻中 human 样本数 */
    @JSONField(name = "cnt_h")
    private Integer cntH;

    /** 近邻中 machine 样本数 */
    @JSONField(name = "cnt_m")
    private Integer cntM;

    /** 预测标签：HUMAN | MACHINE | ERROR */
    @JSONField(name = "label_pred")
    private String labelPred;

    /** 是否低置信度 */
    @JSONField(name = "low_confidence")
    private Boolean lowConfidence;

    // ---- getters & setters ----

    public String getText() {
        return text;
    }

    public void setText(String text) {
        this.text = text;
    }

    public double getDc() {
        return dc;
    }

    public void setDc(double dc) {
        this.dc = dc;
    }

    public Double getPm() {
        return pm;
    }

    public void setPm(Double pm) {
        this.pm = pm;
    }

    public Double getDelta() {
        return delta;
    }

    public void setDelta(Double delta) {
        this.delta = delta;
    }

    public Integer getCntH() {
        return cntH;
    }

    public void setCntH(Integer cntH) {
        this.cntH = cntH;
    }

    public Integer getCntM() {
        return cntM;
    }

    public void setCntM(Integer cntM) {
        this.cntM = cntM;
    }

    public String getLabelPred() {
        return labelPred;
    }

    public void setLabelPred(String labelPred) {
        this.labelPred = labelPred;
    }

    public Boolean getLowConfidence() {
        return lowConfidence;
    }

    public void setLowConfidence(Boolean lowConfidence) {
        this.lowConfidence = lowConfidence;
    }

    @Override
    public String toString() {
        return String.format("PredictionResult{text='%.30s...', d_c=%.4f, label=%s}",
                text, dc, labelPred);
    }
}
