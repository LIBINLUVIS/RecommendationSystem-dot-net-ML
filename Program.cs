
namespace ItemBasedCF
{
    class Program
    {
        static void Main(string[] args)
        {
            // train the model 
            //Training.TrainModel();
           
            //Console.WriteLine("Training completed!");

            // use the model for recommendation

            Recommendation.recommend();

        }
    }
}