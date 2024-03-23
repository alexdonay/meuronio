class Program
{
    static void Main(string[] args)
    {
        // Criando um neurônio com uma taxa de aprendizado de 0.5
        Neuronio neuronio = new Neuronio(0.5);

        // Conjunto de entradas e saídas desejadas para a operação AND
        double[][] entradas = new double[][]
        {
            new double[] {0, 0},
            new double[] {0, 1},
            new double[] {1, 0},
            new double[] {1, 1}
        };

        double[] saidasDesejadas = new double[] { 0, 0, 0, 1 }; // Saídas desejadas para a operação AND

        // Treinamento do neurônio
        for (int epoch = 0; epoch < 100; epoch++)
        {
            for (int i = 0; i < entradas.Length; i++)
            {
                double saidaAtual = neuronio.Train(entradas[i], saidasDesejadas[i]);
            }
        }

        // Após o treinamento, testando o neurônio com as entradas originais
        Console.WriteLine("Testando o neurônio treinado para a operação AND:");
        foreach (var entrada in entradas)
        {
            double output = Math.Round(neuronio.Predict(entrada));
            Console.WriteLine($"Entrada: [{entrada[0]}, {entrada[1]}] - Saída: {output}");
        }
    }
}