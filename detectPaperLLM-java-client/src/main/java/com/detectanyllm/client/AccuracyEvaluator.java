package com.detectanyllm.client;

import com.alibaba.fastjson2.JSON;
import com.alibaba.fastjson2.JSONObject;
import com.alibaba.fastjson2.JSONWriter;
import com.detectanyllm.client.model.PredictResponse;
import com.detectanyllm.client.model.PredictionResult;

import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;

/**
 * DetectAnyLLM 泛化评测工具.
 *
 * <p>
 * 一键流程：
 * <ol>
 * <li>在 dev 集上收集 DC 分数并搜索最优阈值（MCC / F1）</li>
 * <li>将 dev 选出的阈值固定到 test 集上评估</li>
 * <li>在 dev 集按 precision 目标（90% / 95%）选择分级阈值并在 test 验证</li>
 * </ol>
 *
 * <p>
 * 输出文件：
 * <ul>
 * <li>{@code dev_scores.jsonl} / {@code test_scores.jsonl}</li>
 * <li>{@code detailed_metrics.json}</li>
 * <li>{@code risk_thresholds_2tier.json}</li>
 * </ul>
 *
 * <pre>{@code
 *   # 默认: baseUrl=http://117.50.191.32:9000, dev=data/splits_v2/dev.jsonl, test=data/splits_v2/test.jsonl
 *   mvn -f detectPaperLLM-java-client/pom.xml exec:java \
 *       -Dexec.mainClass="com.detectanyllm.client.AccuracyEvaluator"
 *
 *   # 自定义: baseUrl devFile testFile outputDir
 *   mvn exec:java -Dexec.mainClass="com.detectanyllm.client.AccuracyEvaluator" \
 *       -Dexec.args="http://117.50.191.32:9000 /path/to/dev.jsonl /path/to/test.jsonl /path/to/output_dir"
 * }</pre>
 */
public class AccuracyEvaluator {

    private static final String DEFAULT_BASE_URL = "http://117.50.191.32:9000";
    private static final String DEFAULT_DEV_FILE = "data/splits_v2/dev.jsonl";
    private static final String DEFAULT_TEST_FILE = "data/splits_v2/test.jsonl";
    private static final String DEFAULT_OUTPUT_DIR = "detectPaperLLM-java-client/eval_api_dev_test";
    private static final int BATCH_SIZE = 32;
    private static final double MIDDLE_PRECISION_TARGET = 0.90;
    private static final double HIGH_PRECISION_TARGET = 0.95;

    private static final DateTimeFormatter TS_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private enum Objective {
        MCC,
        F1
    }

    private record EvalConfig(String baseUrl, String devFile, String testFile, String outputDir) {
    }

    private record SplitData(List<String> humanTexts, List<String> machineTexts) {
        int totalSamples() {
            return humanTexts.size() + machineTexts.size();
        }
    }

    private record ScorePoint(double score, int label) {
    }

    private record Confusion(int tp, int fp, int tn, int fn) {
    }

    private record Metrics(
            double threshold,
            double aucRoc,
            double mcc,
            double f1,
            double precisionMachine,
            double recallMachine,
            double specificityHuman,
            double npvHuman,
            double accuracy,
            double balancedAccuracy,
            double fpr,
            double fnr,
            Confusion confusion) {
    }

    private record BestThreshold(double threshold, double bestScore, Objective objective) {
    }

    private record PrecisionTargetThreshold(double threshold, Metrics devMetrics) {
    }

    public static void main(String[] args) {
        EvalConfig config = parseArgs(args);
        new AccuracyEvaluator().run(config);
    }

    private static EvalConfig parseArgs(String[] args) {
        String baseUrl = DEFAULT_BASE_URL;
        String devFile = DEFAULT_DEV_FILE;
        String testFile = DEFAULT_TEST_FILE;
        String outputDir = DEFAULT_OUTPUT_DIR;

        // 兼容旧调用：两个参数时按「baseUrl + testFile」处理。
        if (args.length >= 1) {
            baseUrl = args[0];
        }
        if (args.length == 2) {
            testFile = args[1];
        }
        if (args.length >= 3) {
            devFile = args[1];
            testFile = args[2];
        }
        if (args.length >= 4) {
            outputDir = args[3];
        }

        return new EvalConfig(baseUrl, devFile, testFile, outputDir);
    }

    public void run(EvalConfig config) {
        System.out.println("╔═══════════════════════════════════════════════════╗");
        System.out.println("║ DetectAnyLLM 泛化评测 — dev选阈值 + test固定评估 ║");
        System.out.println("╚═══════════════════════════════════════════════════╝");
        System.out.println("API 地址: " + config.baseUrl());
        System.out.println("DEV 文件: " + config.devFile());
        System.out.println("TEST文件: " + config.testFile());
        System.out.println("输出目录: " + config.outputDir());
        System.out.println();

        SplitData devData;
        SplitData testData;
        try {
            devData = loadSplit(Paths.get(config.devFile()));
            testData = loadSplit(Paths.get(config.testFile()));
        } catch (IOException e) {
            System.err.println("❌ 读取数据集失败: " + e.getMessage());
            return;
        }

        System.out.printf("▸ DEV: %d human + %d machine = %d%n",
                devData.humanTexts().size(), devData.machineTexts().size(), devData.totalSamples());
        System.out.printf("▸ TEST: %d human + %d machine = %d%n",
                testData.humanTexts().size(), testData.machineTexts().size(), testData.totalSamples());
        System.out.println();

        try (DetectAnyLLMClient client = new DetectAnyLLMClient(config.baseUrl())) {
            System.out.println("▸ 正在进行健康检查...");
            var health = client.health();
            System.out.printf("  状态: %s | 设备: %s | 后端: %s%n%n",
                    health.getStatus(), health.getDevice(), health.getBackend());

            List<ScorePoint> devScores = evaluateSplit(client, devData, "DEV");
            List<ScorePoint> testScores = evaluateSplit(client, testData, "TEST");

            BestThreshold bestMccDev = findBestThreshold(devScores, Objective.MCC);
            BestThreshold bestF1Dev = findBestThreshold(devScores, Objective.F1);

            Metrics devAtMcc = metricsAtThreshold(devScores, bestMccDev.threshold());
            Metrics devAtF1 = metricsAtThreshold(devScores, bestF1Dev.threshold());
            Metrics testAtMccThreshold = metricsAtThreshold(testScores, bestMccDev.threshold());
            Metrics testAtF1Threshold = metricsAtThreshold(testScores, bestF1Dev.threshold());

            PrecisionTargetThreshold middleRisk = findThresholdForPrecisionTarget(
                    devScores, MIDDLE_PRECISION_TARGET);
            PrecisionTargetThreshold highRisk = findThresholdForPrecisionTarget(
                    devScores, HIGH_PRECISION_TARGET);
            Metrics middleRiskTest = metricsAtThreshold(testScores, middleRisk.threshold());
            Metrics highRiskTest = metricsAtThreshold(testScores, highRisk.threshold());

            writeOutputs(
                    config,
                    devScores,
                    testScores,
                    bestMccDev,
                    bestF1Dev,
                    testAtMccThreshold,
                    testAtF1Threshold,
                    middleRisk,
                    highRisk,
                    middleRiskTest,
                    highRiskTest);

            printSummary(
                    bestMccDev,
                    bestF1Dev,
                    devAtMcc,
                    devAtF1,
                    testAtMccThreshold,
                    testAtF1Threshold,
                    middleRisk,
                    highRisk,
                    middleRiskTest,
                    highRiskTest,
                    config.outputDir());

        } catch (Exception e) {
            System.err.println("❌ 评测过程出错: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private SplitData loadSplit(Path splitPath) throws IOException {
        List<String> humanTexts = new ArrayList<>();
        List<String> machineTexts = new ArrayList<>();
        try (BufferedReader reader = Files.newBufferedReader(splitPath)) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }
                JSONObject obj = JSON.parseObject(line);
                String humanText = obj.getString("human");
                String machineText = obj.getString("machine");
                if (humanText != null && !humanText.isEmpty()) {
                    humanTexts.add(humanText);
                }
                if (machineText != null && !machineText.isEmpty()) {
                    machineTexts.add(machineText);
                }
            }
        }
        return new SplitData(humanTexts, machineTexts);
    }

    private List<ScorePoint> evaluateSplit(DetectAnyLLMClient client, SplitData splitData, String splitName)
            throws IOException {
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.printf("▸ 开始收集 %s 分数%n", splitName);
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        List<ScorePoint> scorePoints = new ArrayList<>(splitData.totalSamples());
        evaluateLabelBatch(client, splitData.humanTexts(), 0, splitName, "HUMAN", scorePoints);
        evaluateLabelBatch(client, splitData.machineTexts(), 1, splitName, "MACHINE", scorePoints);

        System.out.printf("▸ %s 分数收集完成，共 %d 条%n%n", splitName, scorePoints.size());
        return scorePoints;
    }

    private void evaluateLabelBatch(
            DetectAnyLLMClient client,
            List<String> texts,
            int label,
            String splitName,
            String labelName,
            List<ScorePoint> sink) throws IOException {
        int totalBatches = (texts.size() + BATCH_SIZE - 1) / BATCH_SIZE;
        int processed = 0;
        for (int batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
            int start = batchIdx * BATCH_SIZE;
            int end = Math.min(start + BATCH_SIZE, texts.size());
            List<String> batch = texts.subList(start, end);
            System.out.printf("  [%s/%s] 批次 %d/%d (%d 条)...",
                    splitName, labelName, batchIdx + 1, totalBatches, batch.size());

            PredictResponse resp = client.predict(batch);
            List<PredictionResult> predictions = resp.getPredictions();
            if (predictions == null || predictions.size() != batch.size()) {
                throw new IOException(String.format(
                        "%s/%s 响应数量异常: expected=%d actual=%d",
                        splitName, labelName, batch.size(), predictions == null ? 0 : predictions.size()));
            }

            for (PredictionResult result : predictions) {
                sink.add(new ScorePoint(result.getDc(), label));
            }
            processed += batch.size();
            System.out.printf(" 已累计 %d/%d%n", processed, texts.size());
        }
    }

    private BestThreshold findBestThreshold(List<ScorePoint> scores, Objective objective) {
        Set<Double> uniqueThresholds = new TreeSet<>();
        for (ScorePoint p : scores) {
            uniqueThresholds.add(p.score());
        }
        if (uniqueThresholds.isEmpty()) {
            throw new IllegalArgumentException("No scores for threshold search.");
        }

        double bestThreshold = uniqueThresholds.iterator().next();
        double bestScore = Double.NEGATIVE_INFINITY;

        for (double threshold : uniqueThresholds) {
            Confusion confusion = confusionAtThreshold(scores, threshold);
            double current = objective == Objective.MCC
                    ? mccFromConfusion(confusion)
                    : f1FromConfusion(confusion);
            if (current > bestScore) {
                bestScore = current;
                bestThreshold = threshold;
            }
        }
        return new BestThreshold(bestThreshold, bestScore, objective);
    }

    private PrecisionTargetThreshold findThresholdForPrecisionTarget(List<ScorePoint> scores, double targetPrecision) {
        Set<Double> uniqueThresholds = new TreeSet<>();
        for (ScorePoint p : scores) {
            uniqueThresholds.add(p.score());
        }
        if (uniqueThresholds.isEmpty()) {
            throw new IllegalArgumentException("No scores for precision target threshold search.");
        }

        Metrics bestMetrics = null;
        double bestThreshold = 0.0;

        for (double threshold : uniqueThresholds) {
            Metrics metrics = metricsAtThreshold(scores, threshold);
            if (metrics.precisionMachine() + 1e-12 < targetPrecision) {
                continue;
            }
            if (bestMetrics == null
                    || metrics.recallMachine() > bestMetrics.recallMachine()
                    || (Math.abs(metrics.recallMachine() - bestMetrics.recallMachine()) < 1e-12
                            && threshold < bestThreshold)) {
                bestMetrics = metrics;
                bestThreshold = threshold;
            }
        }

        if (bestMetrics == null) {
            throw new IllegalStateException("No threshold can satisfy precision target " + targetPrecision);
        }

        return new PrecisionTargetThreshold(bestThreshold, bestMetrics);
    }

    private Metrics metricsAtThreshold(List<ScorePoint> scores, double threshold) {
        Confusion confusion = confusionAtThreshold(scores, threshold);
        int tp = confusion.tp();
        int fp = confusion.fp();
        int tn = confusion.tn();
        int fn = confusion.fn();

        double precision = safeDiv(tp, tp + fp);
        double recall = safeDiv(tp, tp + fn);
        double specificity = safeDiv(tn, tn + fp);
        double npv = safeDiv(tn, tn + fn);
        double f1 = f1FromConfusion(confusion);
        double accuracy = safeDiv(tp + tn, tp + fp + tn + fn);
        double balancedAccuracy = (recall + specificity) / 2.0;
        double fpr = safeDiv(fp, fp + tn);
        double fnr = safeDiv(fn, fn + tp);
        double mcc = mccFromConfusion(confusion);
        double auc = rocAucFromScores(scores);

        return new Metrics(
                threshold,
                auc,
                mcc,
                f1,
                precision,
                recall,
                specificity,
                npv,
                accuracy,
                balancedAccuracy,
                fpr,
                fnr,
                confusion);
    }

    private Confusion confusionAtThreshold(List<ScorePoint> scores, double threshold) {
        int tp = 0;
        int fp = 0;
        int tn = 0;
        int fn = 0;
        for (ScorePoint p : scores) {
            int pred = p.score() >= threshold ? 1 : 0;
            if (pred == 1 && p.label() == 1) {
                tp++;
            } else if (pred == 1 && p.label() == 0) {
                fp++;
            } else if (pred == 0 && p.label() == 0) {
                tn++;
            } else {
                fn++;
            }
        }
        return new Confusion(tp, fp, tn, fn);
    }

    private double mccFromConfusion(Confusion c) {
        double num = (double) c.tp() * c.tn() - (double) c.fp() * c.fn();
        double den = Math.sqrt(
                (double) (c.tp() + c.fp())
                        * (c.tp() + c.fn())
                        * (c.tn() + c.fp())
                        * (c.tn() + c.fn()));
        return den == 0.0 ? 0.0 : num / den;
    }

    private double f1FromConfusion(Confusion c) {
        return safeDiv(2.0 * c.tp(), 2.0 * c.tp() + c.fp() + c.fn());
    }

    private double rocAucFromScores(List<ScorePoint> scores) {
        int n = scores.size();
        if (n == 0) {
            throw new IllegalArgumentException("scores cannot be empty.");
        }

        int nPos = 0;
        for (ScorePoint p : scores) {
            if (p.label() == 1) {
                nPos++;
            }
        }
        int nNeg = n - nPos;
        if (nPos == 0 || nNeg == 0) {
            return 0.5;
        }

        List<ScorePoint> sorted = new ArrayList<>(scores);
        sorted.sort(Comparator.comparingDouble(ScorePoint::score));

        double sumPosRanks = 0.0;
        int i = 0;
        while (i < sorted.size()) {
            int j = i + 1;
            while (j < sorted.size() && sorted.get(j).score() == sorted.get(i).score()) {
                j++;
            }
            double avgRank = (i + 1 + j) / 2.0;
            int posInTie = 0;
            for (int k = i; k < j; k++) {
                posInTie += sorted.get(k).label();
            }
            sumPosRanks += posInTie * avgRank;
            i = j;
        }

        return (sumPosRanks - (double) nPos * (nPos + 1) / 2.0) / ((double) nPos * nNeg);
    }

    private void writeOutputs(
            EvalConfig config,
            List<ScorePoint> devScores,
            List<ScorePoint> testScores,
            BestThreshold bestMccDev,
            BestThreshold bestF1Dev,
            Metrics testAtMccThreshold,
            Metrics testAtF1Threshold,
            PrecisionTargetThreshold middleRisk,
            PrecisionTargetThreshold highRisk,
            Metrics middleRiskTest,
            Metrics highRiskTest) throws IOException {
        Path outDir = Paths.get(config.outputDir());
        Files.createDirectories(outDir);

        writeScoresJsonl(devScores, outDir.resolve("dev_scores.jsonl"));
        writeScoresJsonl(testScores, outDir.resolve("test_scores.jsonl"));
        writeDetailedMetricsJson(
                config,
                bestMccDev,
                bestF1Dev,
                testAtMccThreshold,
                testAtF1Threshold,
                outDir.resolve("detailed_metrics.json"));
        writeRiskThresholdsJson(
                middleRisk,
                highRisk,
                middleRiskTest,
                highRiskTest,
                outDir.resolve("risk_thresholds_2tier.json"));
    }

    private void writeScoresJsonl(List<ScorePoint> scores, Path file) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(file.toFile()))) {
            for (ScorePoint p : scores) {
                LinkedHashMap<String, Object> row = new LinkedHashMap<>();
                row.put("score", p.score());
                row.put("label", p.label());
                writer.println(JSON.toJSONString(row));
            }
        }
    }

    private void writeDetailedMetricsJson(
            EvalConfig config,
            BestThreshold bestMccDev,
            BestThreshold bestF1Dev,
            Metrics testAtMccThreshold,
            Metrics testAtF1Threshold,
            Path file) throws IOException {
        LinkedHashMap<String, Object> root = new LinkedHashMap<>();

        LinkedHashMap<String, Object> meta = new LinkedHashMap<>();
        meta.put("base_url", config.baseUrl());
        meta.put("dev_file", config.devFile());
        meta.put("test_file", config.testFile());
        meta.put("timestamp", LocalDateTime.now().format(TS_FORMAT));
        root.put("meta", meta);

        LinkedHashMap<String, Object> devThresholds = new LinkedHashMap<>();
        LinkedHashMap<String, Object> mccObj = new LinkedHashMap<>();
        mccObj.put("threshold", bestMccDev.threshold());
        mccObj.put("dev_best_mcc", bestMccDev.bestScore());
        devThresholds.put("mcc_objective", mccObj);

        LinkedHashMap<String, Object> f1Obj = new LinkedHashMap<>();
        f1Obj.put("threshold", bestF1Dev.threshold());
        f1Obj.put("dev_best_f1", bestF1Dev.bestScore());
        devThresholds.put("f1_objective", f1Obj);
        root.put("dev_thresholds", devThresholds);

        root.put("test_metrics_using_dev_mcc_threshold", metricsToMap(testAtMccThreshold));
        root.put("test_metrics_using_dev_f1_threshold", metricsToMap(testAtF1Threshold));

        Files.writeString(file, JSON.toJSONString(root, JSONWriter.Feature.PrettyFormat));
    }

    private void writeRiskThresholdsJson(
            PrecisionTargetThreshold middleRisk,
            PrecisionTargetThreshold highRisk,
            Metrics middleRiskTest,
            Metrics highRiskTest,
            Path file) throws IOException {
        LinkedHashMap<String, Object> root = new LinkedHashMap<>();
        root.put("method", "thresholds selected on dev by precision target, evaluated on test");

        LinkedHashMap<String, Object> targets = new LinkedHashMap<>();
        targets.put("middle_precision_target", MIDDLE_PRECISION_TARGET);
        targets.put("high_precision_target", HIGH_PRECISION_TARGET);
        root.put("targets", targets);

        LinkedHashMap<String, Object> middle = new LinkedHashMap<>();
        middle.put("threshold", middleRisk.threshold());
        middle.put("dev_metrics", metricsToMap(middleRisk.devMetrics()));
        middle.put("test_metrics", metricsToMap(middleRiskTest));
        root.put("middle_risk", middle);

        LinkedHashMap<String, Object> high = new LinkedHashMap<>();
        high.put("threshold", highRisk.threshold());
        high.put("dev_metrics", metricsToMap(highRisk.devMetrics()));
        high.put("test_metrics", metricsToMap(highRiskTest));
        root.put("high_risk", high);

        Files.writeString(file, JSON.toJSONString(root, JSONWriter.Feature.PrettyFormat));
    }

    private LinkedHashMap<String, Object> metricsToMap(Metrics metrics) {
        LinkedHashMap<String, Object> map = new LinkedHashMap<>();
        map.put("threshold", metrics.threshold());
        map.put("auc_roc", metrics.aucRoc());
        map.put("mcc", metrics.mcc());
        map.put("f1", metrics.f1());
        map.put("precision_machine", metrics.precisionMachine());
        map.put("recall_machine", metrics.recallMachine());
        map.put("specificity_human", metrics.specificityHuman());
        map.put("npv_human", metrics.npvHuman());
        map.put("accuracy", metrics.accuracy());
        map.put("balanced_accuracy", metrics.balancedAccuracy());
        map.put("fpr", metrics.fpr());
        map.put("fnr", metrics.fnr());

        LinkedHashMap<String, Object> confusion = new LinkedHashMap<>();
        confusion.put("tp", metrics.confusion().tp());
        confusion.put("fp", metrics.confusion().fp());
        confusion.put("tn", metrics.confusion().tn());
        confusion.put("fn", metrics.confusion().fn());
        map.put("confusion", confusion);
        return map;
    }

    private void printSummary(
            BestThreshold bestMccDev,
            BestThreshold bestF1Dev,
            Metrics devAtMcc,
            Metrics devAtF1,
            Metrics testAtMccThreshold,
            Metrics testAtF1Threshold,
            PrecisionTargetThreshold middleRisk,
            PrecisionTargetThreshold highRisk,
            Metrics middleRiskTest,
            Metrics highRiskTest,
            String outputDir) {
        System.out.println("╔═══════════════════════════════════════════════════╗");
        System.out.println("║                    评测结论                       ║");
        System.out.println("╚═══════════════════════════════════════════════════╝");
        System.out.printf("DEV 最优阈值 (MCC): %.9f | dev_best_mcc=%.6f | dev_f1=%.6f%n",
                bestMccDev.threshold(), bestMccDev.bestScore(), devAtMcc.f1());
        System.out.printf("DEV 最优阈值 (F1) : %.9f | dev_best_f1=%.6f | dev_mcc=%.6f%n%n",
                bestF1Dev.threshold(), bestF1Dev.bestScore(), devAtF1.mcc());

        System.out.println("TEST（使用 dev-MCC 阈值）:");
        printMetricsLine(testAtMccThreshold);
        System.out.println("TEST（使用 dev-F1 阈值）:");
        printMetricsLine(testAtF1Threshold);
        System.out.println();

        System.out.printf("中风险阈值（precision ≥ %.0f%%）: %.9f | test_prec=%.4f | test_rec=%.4f%n",
                MIDDLE_PRECISION_TARGET * 100, middleRisk.threshold(),
                middleRiskTest.precisionMachine(), middleRiskTest.recallMachine());
        System.out.printf("高风险阈值（precision ≥ %.0f%%）: %.9f | test_prec=%.4f | test_rec=%.4f%n",
                HIGH_PRECISION_TARGET * 100, highRisk.threshold(),
                highRiskTest.precisionMachine(), highRiskTest.recallMachine());

        System.out.println();
        System.out.println("▸ 结果文件：");
        System.out.println("  - " + Paths.get(outputDir).resolve("dev_scores.jsonl"));
        System.out.println("  - " + Paths.get(outputDir).resolve("test_scores.jsonl"));
        System.out.println("  - " + Paths.get(outputDir).resolve("detailed_metrics.json"));
        System.out.println("  - " + Paths.get(outputDir).resolve("risk_thresholds_2tier.json"));
        System.out.println();
        System.out.println("✅ 评测完成!");
    }

    private void printMetricsLine(Metrics m) {
        System.out.printf(
                "  threshold=%.9f | AUC=%.6f | MCC=%.6f | F1=%.6f | Acc=%.4f | P=%.4f | R=%.4f | TP=%d FP=%d TN=%d FN=%d%n",
                m.threshold(),
                m.aucRoc(),
                m.mcc(),
                m.f1(),
                m.accuracy(),
                m.precisionMachine(),
                m.recallMachine(),
                m.confusion().tp(),
                m.confusion().fp(),
                m.confusion().tn(),
                m.confusion().fn());
    }

    private double safeDiv(double num, double den) {
        if (den == 0.0) {
            return 0.0;
        }
        return num / den;
    }
}
