using System;
using System.Data;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace ItemBasedCF
{
    public class Rating
    {
        [LoadColumn(0)]
        public float userId;

        [LoadColumn(1)]
        public float movieId;

        [LoadColumn(2)]
        public float rating;

        [LoadColumn(3)]
        public float timestamp;
    }

    public class RatingPrediction
    {
        public float Label;
        public float Score;
    }

    public static class Training
    {
        public static void TrainModel()
        {
            var mlContext = new MLContext();

            // Load data
            IDataView dataView = mlContext.Data.LoadFromTextFile<Rating>("", separatorChar: ',', hasHeader: true);

            // Define the model
            var dataProcessingPipeline = mlContext.Transforms.Conversion.MapValueToKey("UserIdEncoded", "userId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey("ItemIdEncoded", "movieId"));

            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "UserIdEncoded",
                MatrixRowIndexColumnName = "ItemIdEncoded",
                LabelColumnName = nameof(Rating.rating),
                NumberOfIterations = 20,
                ApproximationRank = 100
            };

            var trainingPipeline = dataProcessingPipeline.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

            // Train the model
            var model = trainingPipeline.Fit(dataView);

            // Save the model
            mlContext.Model.Save(model, dataView.Schema, "model.zip");

            Console.WriteLine("Model trained and saved to model.zip");
        }
    }
}
