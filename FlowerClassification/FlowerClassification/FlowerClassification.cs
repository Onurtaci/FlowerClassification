using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace FlowerClassification
{
    public class Neuron
    {
        static Random random = new Random();
        public double[] Weights = new double[4];
        public int ExpectedValue; // Verilerde hangi türdeki bitkinin girdileri ile işlem yapıyorsak o türün beklenen değeri 1 yapılacak.. 
        public string Target; // Bitkinin hangi tür olduğu bilgisi tutuluyor..

        public Neuron()
        {
            for (int i = 0; i < Weights.Length; i++) Weights[i] = random.NextDouble(); // oluşan Neuron nesnesine random ağırlık değerleri atanıyor..
        }

        public double CalculateOutput(double[] inputs)
        {
            double output = 0;
            for (int i = 0; i < inputs.Length; i++) output += inputs[i] * Weights[i]; // Projede istenen nöron çıktısı hesaplama işlemi yapılıyor..
            return output;
        }
    }

    public class NeuralNetwork
    {
        public Neuron[] Neurons { get; set; } = new Neuron[3]; // Yapay sinir ağında bulunması gereken 3 Neuron nesnesi bir dizi içinde tutulacak..
        public double[] Inputs { get; set; } = new double[4];

        public NeuralNetwork()
        {
            for (int i = 0; i < Neurons.Length; i++) Neurons[i] = new Neuron(); // Yapay Sinir Ağı oluşturulduğunda nöronlar için bellekte yer açılıyor.. 
            Neurons[0].Target = "Iris-setosa"; // Neurons dizisinin her bir nöronuna tanıması gereken bitki türü atanıyor..
            Neurons[1].Target = "Iris-versicolor";
            Neurons[2].Target = "Iris-virginica";
        }

        public void Train(List<string[]> dataList, int epochNumber, double lambda) // eğitim yapan metot verileri, epok sayısını ve lambda değerlerini parametre olarak alıyor..
        {
            List<double> outputValues = new List<double>(); // Neurons dizisindeki her bir nöronun hangi çıktı değerini ürettiğini tutacak dizi..
            for (int epochNum = 0; epochNum < epochNumber; epochNum++)
            {
                for (int i = 0; i < dataList.Count; i++)
                {
                    // alttaki for döngüsü ile o an hangi bitki örneğinin girdilerini işleme alıyorsak
                    // o bitkinin türünü tanıyan nöronun beklenen değeri 1, diğer iki nöronun beklenen değeri ise 0 yapılıyor..
                    for (int j = 0; j < Neurons.Length; j++)
                    {
                        Neurons[0].ExpectedValue = 0;
                        Neurons[1].ExpectedValue = 0;
                        Neurons[2].ExpectedValue = 0;
                        if (dataList[i][4] == Neurons[j].Target)
                        { Neurons[j].ExpectedValue = 1; break; }
                    }

                    // alttaki döngüde nöronlar için girdiler atanıyor(Projede istenilen şekilde girdi değerleri 10 a bölünerek atanıyor)..
                    foreach (Neuron neuron in Neurons)
                    {
                        for (int j = 0; j < dataList[i].Length - 1; j++) Inputs[j] = Convert.ToDouble(dataList[i][j]) / 10;
                    }

                    outputValues.Clear(); // her bitki örneği için 3 farklı nöronun çıktısını tutan liste sıfırlanıyor..

                    for (int j = 0; j < Neurons.Length; j++)
                    {
                        outputValues.Add(Neurons[j].CalculateOutput(Inputs)); // her nöronun çıktısı sırasıyla outputValues listesine atılıyor..
                    }

                    int maxValueIndex = outputValues.IndexOf(outputValues.Max()); // maxValueIndex değişkenine hangi indexli nöronun çıktısının en büyük olduğu atanıyor..

                    for (int j = 0; j < Neurons.Length; j++)
                    {
                        if (Neurons[j].ExpectedValue == 1 && maxValueIndex == j) continue; // eğer  beklenen çıktı değeri 1 olan nöron ile ağın ürettiği çıktılardan
                                                                                           // en büyük değere sahip olan nöron aynı ise işlem yapılmıyor ve devam ediliyor
                        if (Neurons[j].ExpectedValue == 1 && maxValueIndex != j)
                        {
                            for (int k = 0; k < Neurons[j].Weights.Length; k++)
                            {
                                // eğer  beklenen çıktı değeri 1 olan nöron ile ağın ürettiği çıktılardan en büyük değere sahip olan nöron farklı ise..
                                Neurons[j].Weights[k] += lambda * Inputs[k];// beklenen çıktı değeri 1 olan nöronun ağırlıklarının değerleri istenilen şekilde arttırılıyor
                                Neurons[maxValueIndex].Weights[k] -= lambda * Inputs[k]; // en büyük değere sahip olan nöronun ağırlıkları ise istenilen şekilde azaltılıyor
                            }
                            break;
                        }
                    }
                }
            }
        }

        public double GetTruthValue(List<string[]> dataList)
        {
            double correctlyClassified = 0; // Doğru sınıflandırılan bitki sayısı tutulacak..
            List<double> outputValues = new List<double>(); // Neurons dizisindeki her bir nöronun hangi çıktı değerini ürettiğini tutacak dizi..
            for (int i = 0; i < dataList.Count; i++)
            {
                // alttaki for döngüsü ile o an hangi bitki örneğinin girdilerini işleme alıyorsak
                // o bitkinin türünü tanıyan nöronun beklenen değeri 1, diğer iki nöronun beklenen değeri ise 0 yapılıyor..
                for (int j = 0; j < Neurons.Length; j++)
                {
                    Neurons[0].ExpectedValue = 0;
                    Neurons[1].ExpectedValue = 0;
                    Neurons[2].ExpectedValue = 0;
                    if (dataList[i][4] == Neurons[j].Target)
                    { Neurons[j].ExpectedValue = 1; break; }
                }
                // alttaki döngüde nöronlara girdiler atanıyor(Projede istenilen şekilde girdi değerleri 10 a bölünerek atanıyor)..
                foreach (Neuron neuron in Neurons)
                {
                    for (int j = 0; j < dataList[i].Length - 1; j++) Inputs[j] = Convert.ToDouble(dataList[i][j]) / 10;
                }

                outputValues.Clear(); // her bitki örneği için 3 farklı nöronun çıktısını tutan liste sıfırlanıyor..

                for (int j = 0; j < Neurons.Length; j++) outputValues.Add(Neurons[j].CalculateOutput(Inputs)); // her nöronun çıktısı sırasıyla outputValues listesine atılıyor..

                int maxValueIndex = outputValues.IndexOf(outputValues.Max()); // maxValueIndex değişkenine hangi indexli nöronun çıktısının en büyük olduğu atanıyor..

                for (int j = 0; j < Neurons.Length; j++)
                {
                    // eğer  beklenen çıktı değeri 1 olan nöron ile ağın ürettiği çıktılardan
                    // en büyük değere sahip olan nöron aynı ise doğru sınıflandırılanların sayısı 1 arttırılıyor..
                    if (Neurons[j].ExpectedValue == 1 && maxValueIndex == j)
                    { correctlyClassified++; break; }
                }
            }
            return 100 * correctlyClassified / (dataList.Count); // Doğruluk değeri yüzde olarak hesaplanıp döndürülüyor..
        }
    }

    internal class FlowerClassification
    {
        static void Main(string[] args)
        {
            List<string[]> dataList = GetDatalist(); // iris.data dosyası okunup içindeki bilgilerin bir listesi döndürülüyor..

            NeuralNetwork[] neuralNetworks = new NeuralNetwork[28];

            for (int i = 0; i < neuralNetworks.Length; i++) neuralNetworks[i] = new NeuralNetwork();

            neuralNetworks[0].Train(dataList, 50, 0.01);

            Console.WriteLine("50 epok ve 0.01 lambda değeri için doğruluk değeri: %" + neuralNetworks[0].GetTruthValue(dataList).ToString("##0.00"));

            int[] epochs = { 20, 50, 100 };
            double[] lambda = { 0.005, 0.01, 0.025 };

            int index = 1;
            for (int i = 1; i < 4; i++)
            {
                Console.WriteLine("\n\n| Deney " + i + "        | 20 Epok | 50 Epok | 100 Epok|");
                foreach (double l in lambda)
                {
                    Console.WriteLine("|----------------------------------------------|");
                    Console.Write("| lambda = " + l.ToString("0.000") + " | ");
                    foreach (int epoch in epochs)
                    {
                        neuralNetworks[index].Train(dataList, epoch, l);
                        Console.Write("%" + neuralNetworks[index++].GetTruthValue(dataList).ToString("##0.00") + "  | ");
                    }
                    Console.WriteLine();
                }

            }

            Console.ReadKey();
        }

        private static List<string[]> GetDatalist()
        {
            // bu metotta iris.data veri dosyası okunup her bir elemanı string[] olan bir liste yapısına atılıyor..
            List<string[]> dataList = new List<string[]>();
            string filePath = @"D:\Ege University\3rd Semester\Data Structures\Projects\Proje-1\FlowerClassification\FlowerClassification\iris.data"; // bu kısım dosya yolu olduğundan başka bir bilgisayarda iris.data hangi 
                                                                                                                  // dosya yolunda ise onun ile değiştirilmesi gerekiyor!!              
            FileStream fs = new FileStream(filePath, FileMode.Open, FileAccess.Read);
            StreamReader sr = new StreamReader(fs);
            string text = sr.ReadLine();

            while (text != null)
            {
                string[] lineArr = text.Split(','); // her bir satırdaki 4 input değeri ve bir tür bilgisi ',' ayracına göre ayrılıp 5 elemanlı bir diziye aktarılıyor..
                dataList.Add(lineArr); //bir üstte oluşturulan 5 elemanlı dizi kullanılacak veri listesine aktarılıyor..
                text = sr.ReadLine();
            }
            sr.Close(); fs.Close();
            return dataList;
        }
    }
}