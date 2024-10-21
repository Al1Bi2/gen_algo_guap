
#include <iostream>
#include <vector>
#include <functional>
#include <fstream>
#include <string>
#include <execution>
#include <algorithm>
#include "../libs/gplot++.h"
#include "genetic.hpp"

template <typename T>
void save(const std::string &filename,  std::vector<std::pair<T,double>> args) {
    std::ofstream os(filename);
    for(auto& arg: args){
        for(auto& x: arg.first.get()){
            os<<x<<" ";
        }
        os<<arg.second<<std::endl;
    }
}

void draw(const std::string &filename){
    Gnuplot plt{};
    std::string result = R"(
set grid
set xrange [-500:500]
set yrange [-500:500]
set zrange [-1000:1000]
set isosamples 100
set samples 100

set contour base



# Define the function
f(x, y) = (-x) * sin(sqrt(abs(x))) + (-y) * sin(sqrt(abs(y)))
set cntrparam levels incremental -1000, 100, 1000  # Define contour steps from -1000 to 1000 with step 100

# Plot the function in 3D and add the contour plot
splot f(x, y) with lines , ")"+filename+R"(" with points  pt 7 ps 1.8)" ;    plt.sendcommand(result);
}
template<typename G>
void gene_print(std::pair<G,double> g){
    std::cout<<"Best: (";
    auto genes = g.first.get();  // Get the gene sequence container

    for (size_t i = 0; i < genes.size(); ++i) {
        std::cout << genes[i];
        if (i != genes.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout<<"):"<<g.second<<std::endl;
}
template<typename G, typename R,typename M,typename C,typename A>
void ui(gen::GA<G,R,M,C,A> &ga){
    ///TODO: add print for N -dimension genotype
    int population_size;
    std::cout<<"Choose population size"<<std::endl;
    std::cin>>population_size;
    ga.set_population_size(population_size);
    ga.fill({-500,-500,-500},{500,500,500});
    ga.calculate_fitness();
    std::cout<<"Initial state"<<std::endl;
    auto best = ga.population.best(gen::reproduction::EXTREMUM::MIN);
    gene_print(best);
    
    auto result = ga.get_population();
    //save<gen::genome::RGA<2>>("result.txt",result);
    //draw("result.txt");

    int n = 0;
    while(true){
        std::cout<<"Enter number of iterations or 0 to exit"<<std::endl;
        std::cin>>n;
        if(n==0) break;
        ga.doit(n);
        best = ga.population.best(gen::reproduction::EXTREMUM::MIN);
        gene_print(best);
        result = ga.get_population();
        //save<gen::genome::RGA<2>>("result.txt",result);
        //draw("result.txt");
        
           
    }
    std::cout<<"Exiting..."<<std::endl;
}

// Сохранение результата одного эксперимента
void save_results(const std::string &filename, int n, double best_avg_result, double avg_of_avg_result) {
    std::ofstream os(filename, std::ios_base::app); // Открываем в режиме добавления
    os << n << " " << best_avg_result << " " << avg_of_avg_result << std::endl;
}                   

// Построение графика через gnuplot
void draw_results(const std::string &filename) {
    Gnuplot plt{};
    std::string cmd = R"(
set grid
set xlabel 'n'
set ylabel 'Fitness'

set title 'Best and Average Fitness vs n'
plot ')" + filename + R"(' using 1:2 with linespoints title 'Best Average Fitness', \
     ')" + filename + R"(' using 1:3 with linespoints title 'Average of Average Fitness'
)";
    plt.sendcommand(cmd);
}

// Функция для одного эксперимента
template<typename G, typename R, typename M, typename C,typename A>
std::pair<double, double> run_experiment(gen::GA<G, R, M, C, A> &ga, int epch) {
    // Запуск ГА на epch шагов
    ga.doit(epch);
    
    // Получение лучшего результата
    auto best_pair = ga.population.best(gen::reproduction::EXTREMUM::MIN);
    double best_fitness = best_pair.second;

    // Вычисление среднего значения по всей популяции
    double avg_fitness = 0.0;
    for (const auto &individual : ga.get_population()) {
        avg_fitness += individual.second;
    }
    avg_fitness /= ga.get_population().size();  // Среднее значение популяции

    return {best_fitness, avg_fitness};
}

// Основная функция для многократного запуска экспериментов с разными n
void run_experiments(int epch, const std::vector<int> &n_values, const std::string &output_filename, const int population_size) {
    auto schwefel =std::function( [](const std::vector<double> &x) -> double {
        double sum = 0.0;
        const double constant = 418.9829;
        for (double xi : x) {
            sum -= xi * sin(sqrt(fabs(xi)));
        }
        return sum;
    });

    std::ofstream os(output_filename);
    os.close();
    
    // Перебираем разные значения n
    for (int n : n_values) {
        double total_best = 0.0;  // Для усреднения лучших значений
        double total_avg = 0.0;   // Для усреднения средних значений

        const int iterations = 100;  // Количество запусков эксперимента для усреднения
    const double extremum = 2 * 418.9829;
        // Многократные запуски для усреднения
        for (int i = 0; i < iterations; ++i) {
            gen::GA ga(gen::genome::RGA<2>(),
                   gen::reproduction::roulette{},
                   gen::mutation::non_uniform{5,epch},
                   gen::crossover::RGA::SBX{n},
                   schwefel,
                   gen::reproduction::EXTREMUM::MIN);
            ga.set_population_size(population_size);
            ga.fill({-500, -500}, {500, 500});
            ga.calculate_fitness();

            // Получаем пару значений: лучший результат и среднее значение популяции
            auto [best, avg] = run_experiment(ga, epch);

            total_best += ((best+extremum)*(best+extremum)/iterations);  // Суммируем лучшие результаты
            total_avg += ((avg+extremum)*(avg+extremum)/iterations);    // Суммируем средние значения популяции
        }
        //MSE
        // Сохранение результатов
        save_results(output_filename, n, sqrt(total_best), sqrt(total_avg));
    }
}



int main(){
    std::function<double(std::vector<double>)> schwefel = [](const std::vector<double>& x) -> double {
        double sum = 0.0;
        const double constant = 418.9829;
        for (double xi : x) {
            sum -= xi * sin(sqrt(fabs(xi)));
        }
        return sum;
    };
    std::cout<<"Menu:"<<std::endl;
    std::cout<<"\t1)Steps"<<std::endl;
    std::cout<<"\t2)Experiment"<<std::endl;
    std::cout<<"\t3)Exit"<<std::endl;
    int menu = 0;
    int n =0;
    std::cin>>menu;
    switch(menu){
        case 1:{
             n = 2;
            std::cout<<"Choose N value for SBX "<<std::endl;
            std::cin>>n;
            gen::GA ga(gen::genome::RGA<3>(),gen::reproduction::roulette{},gen::mutation::non_uniform{5,100},gen::crossover::RGA::SBX{n},schwefel,gen::reproduction::EXTREMUM::MIN);
            ga.set_algo(schwefel);
            ui(ga);
        }
            break;
        case 2:{
            int epch = 0;  // Количество поколений для каждого эксперимента
            std::cout<<"Choose number of epochs"<<std::endl;
            std::cin>>epch;
            std::cout<<"choose population size"<<std::endl;
            std::cin>>n;
            
            std::vector<int> n_values = {2, 3, 4, 5};  // Значения n
            std::string output_filename = "fitness_results.txt";  // Файл для сохранения результатов
            run_experiments(epch, n_values, "fitness_results.txt", n);
           // draw_results(output_filename);
        }
            break;
        case 3:
            return 0;
    }
    
    return 0;
}