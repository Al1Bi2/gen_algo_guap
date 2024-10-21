
#include <iostream>
#include <vector>
#include <functional>
#include <fstream>
#include <string>
#include <execution>
#include <algorithm>
#include "../libs/gplot++.h"
#include "../lab2/genetic.hpp"

int main(){
     std::vector<std::vector<double>> dist = {{0,2},{2,0}};
    auto a = [](const std::vector<int>& a, std::vector<std::vector<double>>& dist){
        double sum;
        int next = 0;
        int cur = 0;
        for(int i = 0; i< a.size();i++){
            int next = a[cur];
            sum+=dist[cur][next];
            cur = next;
        }
        return sum;
    };
    auto func =  std::function(a);
    gen::GA ga(gen::genome::salesman::neighbour{dist}, gen::reproduction::roulette{},
            gen::mutation::change{},gen::crossover::salesman::heuristic{dist},func);
    
   ga.doit();
}
