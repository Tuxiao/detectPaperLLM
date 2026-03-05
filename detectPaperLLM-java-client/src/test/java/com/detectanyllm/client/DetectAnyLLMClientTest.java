package com.detectanyllm.client;

import com.detectanyllm.client.model.HealthResponse;
import com.detectanyllm.client.model.PredictRequest;
import com.detectanyllm.client.model.PredictResponse;
import com.detectanyllm.client.model.PredictionResult;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;
import org.junit.jupiter.api.*;

import java.io.IOException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DetectAnyLLMClient 单元测试 — 使用 MockWebServer 模拟 API.
 */
class DetectAnyLLMClientTest {

    private MockWebServer mockServer;
    private DetectAnyLLMClient client;

    @BeforeEach
    void setUp() throws IOException {
        mockServer = new MockWebServer();
        mockServer.start();
        client = new DetectAnyLLMClient(mockServer.url("/").toString());
    }

    @AfterEach
    void tearDown() throws IOException {
        client.close();
        mockServer.shutdown();
    }

    @Test
    @DisplayName("GET /health — 正常返回健康状态")
    void testHealth() throws Exception {
        String body = """
                {
                    "status": "healthy",
                    "model_path": "/root/detectPaperLLM/models",
                    "decision_mode": "threshold",
                    "device": "cuda:0",
                    "backend": "pytorch"
                }
                """;
        mockServer.enqueue(new MockResponse().setBody(body)
                .setHeader("Content-Type", "application/json"));

        HealthResponse health = client.health();

        assertTrue(health.isHealthy());
        assertEquals("cuda:0", health.getDevice());
        assertEquals("pytorch", health.getBackend());

        RecordedRequest req = mockServer.takeRequest();
        assertEquals("GET", req.getMethod());
        assertEquals("/health", req.getPath());
    }

    @Test
    @DisplayName("POST /predict — 单文本检测")
    void testPredictSingle() throws Exception {
        String body = """
                {
                    "predictions": [{
                        "text": "Test sentence.",
                        "d_c": 0.714,
                        "p_m": null,
                        "delta": null,
                        "cnt_h": null,
                        "cnt_m": null,
                        "label_pred": "HUMAN",
                        "low_confidence": null
                    }],
                    "elapsed_ms": 1500.0,
                    "decision_mode": "threshold"
                }
                """;
        mockServer.enqueue(new MockResponse().setBody(body)
                .setHeader("Content-Type", "application/json"));

        PredictResponse resp = client.predict("Test sentence.");

        assertEquals(1, resp.getPredictions().size());
        PredictionResult result = resp.getPredictions().get(0);
        assertEquals("HUMAN", result.getLabelPred());
        assertEquals(0.714, result.getDc(), 0.001);
        assertEquals("threshold", resp.getDecisionMode());

        RecordedRequest req = mockServer.takeRequest();
        assertEquals("POST", req.getMethod());
        assertEquals("/predict", req.getPath());

        // 验证请求 body 包含 text 字段
        String reqBody = req.getBody().readUtf8();
        assertTrue(reqBody.contains("\"text\""));
    }

    @Test
    @DisplayName("POST /predict — 批量检测")
    void testPredictBatch() throws Exception {
        String body = """
                {
                    "predictions": [
                        {"text": "A", "d_c": 0.5, "label_pred": "HUMAN"},
                        {"text": "B", "d_c": 80.0, "label_pred": "MACHINE"}
                    ],
                    "elapsed_ms": 3200.0,
                    "decision_mode": "threshold"
                }
                """;
        mockServer.enqueue(new MockResponse().setBody(body)
                .setHeader("Content-Type", "application/json"));

        PredictResponse resp = client.predict(List.of("A", "B"));

        assertEquals(2, resp.getPredictions().size());
        assertEquals("HUMAN", resp.getPredictions().get(0).getLabelPred());
        assertEquals("MACHINE", resp.getPredictions().get(1).getLabelPred());
    }

    @Test
    @DisplayName("POST /predict — 带自定义参数")
    void testPredictWithOverrides() throws Exception {
        String body = """
                {
                    "predictions": [{"text": "X", "d_c": 1.0, "label_pred": "HUMAN"}],
                    "elapsed_ms": 100.0,
                    "decision_mode": "threshold"
                }
                """;
        mockServer.enqueue(new MockResponse().setBody(body)
                .setHeader("Content-Type", "application/json"));

        PredictRequest req = new PredictRequest("X");
        req.setThreshold(75.0);
        req.setMaxLength(256);
        req.setDecisionMode("threshold");

        PredictResponse resp = client.predict(req);
        assertNotNull(resp);

        // 验证请求中包含自定义参数
        RecordedRequest recordedReq = mockServer.takeRequest();
        String reqBody = recordedReq.getBody().readUtf8();
        assertTrue(reqBody.contains("\"threshold\""));
        assertTrue(reqBody.contains("75.0") || reqBody.contains("75"));
    }

    @Test
    @DisplayName("HTTP 错误 — 抛出 IOException")
    void testHttpError() {
        mockServer.enqueue(new MockResponse().setResponseCode(503)
                .setBody("{\"detail\":\"Model not loaded\"}"));

        IOException ex = assertThrows(IOException.class, () -> client.health());
        assertTrue(ex.getMessage().contains("503"));
    }
}
