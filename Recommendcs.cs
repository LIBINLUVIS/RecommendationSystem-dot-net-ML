using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ItemBasedCF
{
  public static class Recommendation
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

        public static void recommend()
        {
            var mlContext = new MLContext();

            ITransformer loadedModel;
            DataViewSchema modelSchema;

            using (var fileStream = new FileStream("model.zip", FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(fileStream, out modelSchema);
            }

            var predictionEngine = mlContext.Model.CreatePredictionEngine<Rating, RatingPrediction>(loadedModel);

            float userId = 10;

            var dataView = mlContext.Data.LoadFromTextFile<Rating>("", separatorChar: ',', hasHeader: true);

            var items = dataView.GetColumn<float>("movieId").Distinct().ToList();

            var userRatings = mlContext.Data.CreateEnumerable<Rating>(dataView, reuseRowObject: false)
                             .Where(r => r.userId == userId)
                             .Select(r => r.movieId)
                             .ToHashSet();

            var predictions = new List<Tuple<float, float>>();

            foreach (var item in items)
            {
                if (!userRatings.Contains(item))
                {
                    var input = new Rating { userId = userId, movieId = item };
                    var prediction = predictionEngine.Predict(input);
                    predictions.Add(new Tuple<float, float>(item, prediction.Score));
                }
            }

            var top5Recommendations = predictions.OrderByDescending(p => p.Item2).Take(5);

            foreach (var recommendation in top5Recommendations)
            {
                Console.WriteLine("Item: {0}, Predicted Rating: {1}", recommendation.Item1, recommendation.Item2);
            }



        }
    }
}
