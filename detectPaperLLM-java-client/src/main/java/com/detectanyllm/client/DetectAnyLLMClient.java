package com.detectanyllm.client;

import com.alibaba.fastjson2.JSON;
import com.detectanyllm.client.model.*;
import okhttp3.*;

import java.io.Closeable;
import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

/**
 * DetectAnyLLM API 客户端.
 *
 * <p>
 * 使用 OkHttp 发送 HTTP 请求，Fastjson2 处理 JSON 序列化与反序列化。
 * 用完后请调用 {@link #close()} 释放连接池资源。
 *
 * <pre>{@code
 * try (var client = new DetectAnyLLMClient("http://117.50.191.32:9000")) {
 *     PredictResponse resp = client.predict("这是一段测试文本。");
 *     resp.getPredictions().forEach(p -> System.out.printf("标签: %s, DC: %.4f%n", p.getLabelPred(), p.getDc()));
 * }
 * }</pre>
 */
public class DetectAnyLLMClient implements Closeable {

    private static final MediaType JSON_MEDIA_TYPE = MediaType.parse("application/json; charset=utf-8");

    private final String baseUrl;
    private final OkHttpClient httpClient;

    /**
     * 使用默认超时配置创建客户端.
     *
     * @param baseUrl API 基础地址，例如 {@code http://117.50.191.32:9000}
     */
    public DetectAnyLLMClient(String baseUrl) {
        this(baseUrl, new OkHttpClient.Builder()
                .connectTimeout(10, TimeUnit.SECONDS)
                .readTimeout(300, TimeUnit.SECONDS) // 推理可能较慢
                .writeTimeout(30, TimeUnit.SECONDS)
                .build());
    }

    /**
     * 使用自定义 OkHttpClient 创建客户端.
     *
     * @param baseUrl    API 基础地址
     * @param httpClient 自定义 OkHttpClient 实例
     */
    public DetectAnyLLMClient(String baseUrl, OkHttpClient httpClient) {
        this.baseUrl = Objects.requireNonNull(baseUrl).replaceAll("/+$", "");
        this.httpClient = Objects.requireNonNull(httpClient);
    }

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    /**
     * 健康检查.
     *
     * @return 健康状态信息
     * @throws IOException 网络异常
     */
    public HealthResponse health() throws IOException {
        Request request = new Request.Builder()
                .url(baseUrl + "/health")
                .get()
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            return parseResponse(response, HealthResponse.class);
        }
    }

    /**
     * 单文本 AI 痕迹检测.
     *
     * @param text 待检测文本
     * @return 预测响应
     * @throws IOException 网络异常
     */
    public PredictResponse predict(String text) throws IOException {
        return predict(new PredictRequest(text));
    }

    /**
     * 批量文本 AI 痕迹检测.
     *
     * @param texts 待检测文本列表（最多 32 条）
     * @return 预测响应
     * @throws IOException 网络异常
     */
    public PredictResponse predict(List<String> texts) throws IOException {
        return predict(new PredictRequest(texts));
    }

    /**
     * 使用完整请求参数进行检测（支持覆盖阈值、决策模式等）.
     *
     * @param predictRequest 请求对象
     * @return 预测响应
     * @throws IOException 网络异常
     */
    public PredictResponse predict(PredictRequest predictRequest) throws IOException {
        String json = JSON.toJSONString(predictRequest);

        RequestBody body = RequestBody.create(json, JSON_MEDIA_TYPE);
        Request request = new Request.Builder()
                .url(baseUrl + "/predict")
                .post(body)
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            return parseResponse(response, PredictResponse.class);
        }
    }

    @Override
    public void close() {
        httpClient.dispatcher().executorService().shutdown();
        httpClient.connectionPool().evictAll();
    }

    // -----------------------------------------------------------------------
    // Internal
    // -----------------------------------------------------------------------

    private <T> T parseResponse(Response response, Class<T> type) throws IOException {
        ResponseBody responseBody = response.body();
        if (responseBody == null) {
            throw new IOException("Empty response body, HTTP " + response.code());
        }

        String bodyStr = responseBody.string();

        if (!response.isSuccessful()) {
            throw new IOException(String.format(
                    "API request failed: HTTP %d — %s", response.code(), bodyStr));
        }

        return JSON.parseObject(bodyStr, type);
    }
}
