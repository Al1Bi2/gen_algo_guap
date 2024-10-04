
#include <iostream>
#include <vector>
#include <functional>
#include <fstream>
#include <string>
#include <execution>
#include <algorithm>
#include "../libs/gplot++.h"
#include "genetic.hpp"



int main(){
    auto schwefel = [](const std::vector<double>& x) -> double {
    double sum = 0.0;
    const double constant = 418.9829;
    for (double xi : x) {
        sum += xi * sin(sqrt(fabs(xi)));
    }
    return sum;
    };
    gen::GA ga(gen::genome::RGA<2>(),gen::reproduction::tournament{3},gen::mutation::non_uniform{0.5,-500,500},gen::crossover::RGA::SBX{});
    ga.set_popuation_size(20);
    ga.fill({-500,-500},{500,500});
    ga.set_algo(schwefel);
    ga.doit(10);
    auto best = ga.population.best();
     ga.doit(10);
    best = ga.population.best();
    return 0;
}