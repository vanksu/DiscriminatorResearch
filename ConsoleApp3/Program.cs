using System;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Drawing;
using System.Collections.Generic;

class BagDiscriminatorEvaluation
{
    static void Main()
    {
        // Конфигурация
        string autoencoderPath = @"C:\Users\brill\source\autoencoder_full.onnx";
        string discriminatorPath = @"C:\Users\brill\source\bag_discriminator.onnx";
        string datasetPath = @"C:\Users\brill\source\mnist259_64";
        string outputRestoredPath = @"C:\Users\brill\source\restored_images";
        string reportPath = @"C:\Users\brill\source\bag_evaluation_report.csv";

        int targetClassIdx = 8; // "Bag"
        string targetClassName = "Bag";

        // Инициализация моделей
        using var autoencoderSession = new InferenceSession(autoencoderPath);
        using var discriminatorSession = new InferenceSession(discriminatorPath);
        Directory.CreateDirectory(outputRestoredPath);

        // Две отдельные статистики
        var originalStats = new BinaryEvaluationStats();
        var restoredStats = new BinaryEvaluationStats();
        var reportLines = new List<string> {
            "ImagePath,IsBag,OriginalPred,OriginalProb,RestoredPred,RestoredProb"
        };

        // Обработка данных
        for (int classIdx = 0; classIdx < 10; classIdx++)
        {
            string classFolder = Path.Combine(datasetPath, classIdx.ToString());
            if (!Directory.Exists(classFolder)) continue;

            foreach (var imagePath in Directory.GetFiles(classFolder, "*.png"))
            {
                try
                {
                    bool isBag = (classIdx == targetClassIdx);
                    var originalImage = LoadImageAsGrayscale(imagePath);

                    // 1. Оценка оригинала
                    var originalProb = EvaluateBag(discriminatorSession, originalImage);
                    bool originalPred = originalProb > 0.5f;
                    originalStats.Update(isBag, originalPred);

                    // 2. Оценка восстановленного
                    var restoredImage = RestoreImage(autoencoderSession, originalImage);
                    restoredImage.Save(Path.Combine(outputRestoredPath, $"{classIdx}_{Path.GetFileName(imagePath)}"));

                    var restoredProb = EvaluateBag(discriminatorSession, restoredImage);
                    bool restoredPred = restoredProb > 0.5f;
                    restoredStats.Update(isBag, restoredPred);

                    // Запись в отчет
                    reportLines.Add($"{imagePath},{isBag},{originalPred},{originalProb:F4},{restoredPred},{restoredProb:F4}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error processing {imagePath}: {ex.Message}");
                }
            }
        }

        // Сохранение и вывод результатов
        File.WriteAllLines(reportPath, reportLines);

        Console.WriteLine("=== Original Images Evaluation ===");
        originalStats.PrintResults(targetClassName);

        Console.WriteLine("\n=== Restored Images Evaluation ===");
        restoredStats.PrintResults(targetClassName);

        Console.WriteLine($"\nReport saved to: {reportPath}");
    }

    // Методы обработки изображений (без изменений)
    // Методы обработки изображений
    static Bitmap LoadImageAsGrayscale(string path)
    {
        using var original = new Bitmap(path);
        var grayscale = new Bitmap(original.Width, original.Height);

        for (int y = 0; y < original.Height; y++)
            for (int x = 0; x < original.Width; x++)
            {
                var pixel = original.GetPixel(x, y);
                int grayValue = (int)(pixel.R * 0.299 + pixel.G * 0.587 + pixel.B * 0.114);
                grayscale.SetPixel(x, y, Color.FromArgb(grayValue, grayValue, grayValue));
            }
        return grayscale;
    }
    static Bitmap RestoreImage(InferenceSession autoencoder, Bitmap original)
    {
        // Конвертация в 3 канала (RGB)
        var inputTensor = new DenseTensor<float>(new[] { 1, 3, original.Height, original.Width });

        for (int y = 0; y < original.Height; y++)
            for (int x = 0; x < original.Width; x++)
            {
                float pixelValue = original.GetPixel(x, y).R / 255f;
                inputTensor[0, 0, y, x] = pixelValue; // R
                inputTensor[0, 1, y, x] = pixelValue; // G
                inputTensor[0, 2, y, x] = pixelValue; // B
            }

        var inputs = new[] { NamedOnnxValue.CreateFromTensor("input", inputTensor) };

        using var results = autoencoder.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Конвертация обратно в grayscale
        var restored = new Bitmap(original.Width, original.Height);
        for (int y = 0; y < original.Height; y++)
            for (int x = 0; x < original.Width; x++)
            {
                float r = Math.Clamp(outputTensor[0, 0, y, x] * 255, 0, 255);
                float g = Math.Clamp(outputTensor[0, 1, y, x] * 255, 0, 255);
                float b = Math.Clamp(outputTensor[0, 2, y, x] * 255, 0, 255);
                int grayValue = (int)(r * 0.299 + g * 0.587 + b * 0.114);
                restored.SetPixel(x, y, Color.FromArgb(grayValue, grayValue, grayValue));
            }
        return restored;
    }

    static float EvaluateBag(InferenceSession discriminator, Bitmap image)
    {
        var inputTensor = new DenseTensor<float>(new[] { 1, 1, image.Height, image.Width });

        for (int y = 0; y < image.Height; y++)
            for (int x = 0; x < image.Width; x++)
                inputTensor[0, 0, y, x] = image.GetPixel(x, y).R / 255f * 2 - 1; // Нормализация [-1, 1]

        var inputs = new[] { NamedOnnxValue.CreateFromTensor("input", inputTensor) };

        using var results = discriminator.Run(inputs);
        return results.First().AsTensor<float>()[0]; // Вероятность принадлежности к классу

    }
}

// Обновленный класс статистики
class BinaryEvaluationStats
{
    private int truePositives = 0;
    private int trueNegatives = 0;
    private int falsePositives = 0;
    private int falseNegatives = 0;

    public void Update(bool isTargetClass, bool predictedPositive)
    {
        if (isTargetClass)
        {
            if (predictedPositive) truePositives++;
            else falseNegatives++;
        }
        else
        {
            if (predictedPositive) falsePositives++;
            else trueNegatives++;
        }
    }

    public void PrintResults(string className)
    {
        int total = truePositives + trueNegatives + falsePositives + falseNegatives;
        Console.WriteLine($"Accuracy: {(float)(truePositives + trueNegatives) / total:P2}");
        Console.WriteLine($"Precision: {(float)truePositives / (truePositives + falsePositives):P2}");
        Console.WriteLine($"Recall: {(float)truePositives / (truePositives + falseNegatives):P2}");

        Console.WriteLine("\nConfusion Matrix:");
        Console.WriteLine($"| Actual\\Pred | {className} | Not {className} |");
        Console.WriteLine($"|-------------|------------|----------------|");
        Console.WriteLine($"| {className}    | {truePositives,10} | {falseNegatives,14} |");
        Console.WriteLine($"| Not {className} | {falsePositives,10} | {trueNegatives,14} |");
    }
}