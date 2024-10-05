
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


int main(){
    auto schwefel = [](const std::vector<double>& x) -> double {
        double sum = 0.0;
        const double constant = 418.9829;
        for (double xi : x) {
            sum -= xi * sin(sqrt(fabs(xi)));
        }
        return sum;
    };
     auto dejongs = [](const std::vector<double>& x) -> double {
    double sum = 0.0;
    for (double xi : x) {
        sum += xi*xi ;
    }
    return sum;
    };
    gen::GA ga(gen::genome::RGA<2>(),gen::reproduction::roulette{},gen::mutation::non_uniform{0.5,-500,500},gen::crossover::RGA::SBX{5},gen::reproduction::EXTREMUM::MIN);
    ga.set_population_size(10);
    ga.fill({-500,-500},{500,500});
    ga.set_algo(schwefel);

    ga.calculate_fitness();
        auto result = ga.get_population();
        save<gen::genome::RGA<2>>("result.txt",result);
        draw("result.txt");
    ga.doit(1);
        result = ga.get_population();
        save<gen::genome::RGA<2>>("result.txt",result);
        draw("result.txt");
    auto best = ga.population.best(gen::reproduction::EXTREMUM::MIN);
     ga.doit(50);
    best = ga.population.best(gen::reproduction::EXTREMUM::MIN);
        result = ga.get_population();
        save<gen::genome::RGA<2>>("result.txt",result);
        draw("result.txt");
    return 0;
}