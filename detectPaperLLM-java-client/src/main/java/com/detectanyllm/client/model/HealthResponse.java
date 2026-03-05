package com.detectanyllm.client.model;

import com.alibaba.fastjson2.annotation.JSONField;

/**
 * 健康检查响应 DTO — 对应 API GET /health.
 */
public class HealthResponse {

    private String status;

    @JSONField(name = "model_path")
    private String modelPath;

    @JSONField(name = "decision_mode")
    private String decisionMode;

    private String device;

    private String backend;

    // ---- getters & setters ----

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public String getModelPath() {
        return modelPath;
    }

    public void setModelPath(String modelPath) {
        this.modelPath = modelPath;
    }

    public String getDecisionMode() {
        return decisionMode;
    }

    public void setDecisionMode(String decisionMode) {
        this.decisionMode = decisionMode;
    }

    public String getDevice() {
        return device;
    }

    public void setDevice(String device) {
        this.device = device;
    }

    public String getBackend() {
        return backend;
    }

    public void setBackend(String backend) {
        this.backend = backend;
    }

    public boolean isHealthy() {
        return "healthy".equalsIgnoreCase(status);
    }

    @Override
    public String toString() {
        return String.format("HealthResponse{status='%s', device='%s', backend='%s'}",
                status, device, backend);
    }
}
