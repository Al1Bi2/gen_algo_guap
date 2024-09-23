#include <bitset>   
#include <random>
#include <cmath>
#include <iostream>
#include <functional>
#include <iterator>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <chrono>
#include <fstream>
#include "../libs/gplot++.h"

using namespace std;

random_device rnd;
mt19937 gen(rnd());

enum class OptimizationType {
    MINIMIZATION,
    MAXIMIZATION
};


double calculate_fitness(double value, OptimizationType optimization_type, std::function<double(double)> function) {
    double fitness = function(value);
    if (optimization_type == OptimizationType::MINIMIZATION) {
        return 1.0 / (fitness + 1e-10);  // Adding a small value to avoid division by zero
    } else {
        return fitness;
    }
}
 

long random(long min, long max){
    std::uniform_int_distribution<long> distrib(min, max);
    return distrib(gen);
}
double randomd(double min, double max){
    std::uniform_real_distribution<double> distrib(min, max);
    return distrib(gen);
}



template<int min, int max,int dad>
class Population{
public:

    constexpr Population(){}

    
    double get_best_point(){
        auto sort_func = [this](Genome a, Genome b) {return  function(pos_to_real(a.getGenome())) > function(pos_to_real(b.getGenome())); };
        vector<Genome> sorted_pop = population;
        
        sort(sorted_pop.begin(),sorted_pop.end(), sort_func);  
        return pos_to_real(sorted_pop.front().getGenome());
    }

    void setFunction(function<double(double)> fun){
        function = fun;
    }

    void fill( int count){
        assert(count%2 == 0);
        for(int i = 0;i < count; i++){
            Genome g;
            g.setGenome(generate());
            population.push_back(g);
        }
    }
    
    void step(){
            reproduction();
            crossingover();
            mutation();
            //printp();
    }

    void doSteps(int n){
        for(int i=0;i<n;i++){
            step();
        }
    }

    void draw(){
        Gnuplot plt{};
        std::vector<double> x,y;
        for(double i = min; i<max;i+=0.001){
            x.push_back(i);
            y.push_back(function(i)); 
        }

        std::vector<double> px,py;
        for(auto pop: population){
            px.push_back(pos_to_real(pop.gene.to_ulong()));
            py.push_back(function(px.back()));
        }

        plt.plot(x, y, "Dataset #1", Gnuplot::LineStyle::LINES);
        plt.plot(px, py, "Dataset #2", Gnuplot::LineStyle::POINTS);

        plt.show();
    }

    void setPCross(double p){probability_crossingover = p;}
    void setPMut(double p){probability_mutation = p;}

private:
    class Genome;
    static constexpr long lenght(){return static_cast<long>(log2((max-min)*pow(10,dad)))+1;};
    static constexpr long n(){return static_cast<long>(powl(2,lenght()));}

    OptimizationType optimization_type = OptimizationType::MAXIMIZATION;

    std::function<double(double)> function;

    vector<Genome>  population = {};
    vector<Genome> mid_population  ={};

    //Gnuplot plt{};

    double probability_crossingover = 0.5d;
    double probability_mutation = 0.01d;
private:

    double pos_to_real(long pos){
        return pos*(max-min)/static_cast<double>(n()-1)+min;
    }

    long generate(){
        return random(0,n()-1);
    }
    void reproduction(){
        vector<double> fitness  ={};
        vector<double> probabilities  ={};

        double fitness_min = numeric_limits<double>::max();

        for(auto gene: population){
            double fit = calculate_fitness(pos_to_real(gene.getGenome()), optimization_type, function);
            fitness.push_back(fit);
            if (fit < fitness_min) {
                fitness_min = fit;
            }
        }

        double normalized_min = fitness_min < 0 ? -fitness_min : 0;
        for (auto& fit : fitness) {
            fit += normalized_min;
        }

        double total_fitness = accumulate(fitness.begin(), fitness.end(), 0.0);

        for(auto fit: fitness){
            probabilities.push_back(fit/total_fitness);
        }

        mid_population.clear();

        vector<double> cumulative_probabilities(probabilities.size());
        partial_sum(probabilities.begin(), probabilities.end(), cumulative_probabilities.begin());

        for(int i = 0; i< population.size();i++){
            double random_value = randomd(0.0,1.0);
            auto it = lower_bound(cumulative_probabilities.begin(), cumulative_probabilities.end(), random_value);
            size_t index = distance(cumulative_probabilities.begin(), it);
            mid_population.push_back(population[index]);
 
        }   
    }

    void crossingover(){

        
        vector<Genome> a_half  ={};
        vector<Genome> b_half  ={};
        long a,b;
        while(mid_population.size()>0){
            do{
                a = random(0,mid_population.size()-1);
                b = random(0,mid_population.size()-1);
            }while(a==b);
            auto a_it = std::next(mid_population.begin(), a);
            auto b_it = std::next(mid_population.begin(), b);
            a_half.push_back(*a_it);
            b_half.push_back(*b_it);
            mid_population.erase(a_it);
            mid_population.erase(b_it);
        }
        //a_half.push_back(mid_population[0]);
        //b_half.push_back(mid_population[1]);
        for(int i = 0; i < a_half.size();i++){
            long idx =  random(1,lenght()-1);
            double rand_v  = randomd(0.0d, 1.0d);
            if(rand_v < probability_crossingover){
                a_half[i].swap(b_half[i],idx);
            }     
            
        }
        mid_population.insert(mid_population.end(),a_half.begin(),a_half.end());
        mid_population.insert(mid_population.end(),b_half.begin(),b_half.end());
       
    }

    void mutation(){
        for(auto& pop: mid_population){
            double rand_v = randomd(0.0d, 1.0d);
            if(rand_v < probability_mutation){
                long idx =  random(0,lenght()-1);

                pop.gene.flip(idx);
            }
        }

        population.clear();
        for(auto pop: mid_population){
            population.push_back(pop);
        }
        //mid_population.clear();
    }



private:

    class  Genome{
    public:
        bitset<lenght()> gene;

        void setGenome(unsigned long value){
            gene = bitset<lenght()>(value);
            
        }
        long getGenome(){
            return gene.to_ulong();
        }
        void flip(int idx){
            gene.flip(idx);
        }
        void swap(Genome& other, int end_idx){
            for(size_t i = 0; i < end_idx; i++){
            bool tmp = gene[i];
            this->gene[i] = other.gene[i];
            other.gene[i] = tmp;  
            }
        }
    };
public:
    
    void printp(){
        for(auto p: population){
            cout<<p.getGenome()<<" ";
        }
        cout<<endl;;
    }
    void printm(){
        for(auto p: mid_population){
            cout<<p.gene.to_ulong()<<" ";
        }
        cout<<endl;
    } 
};
vector<int> generate_population_range(int min, int max) {
    vector<int> populations;
    for (int i = min; i <= max; i += 2) { 
        populations.push_back(i);
    }
    return populations;
}


vector<double> generate_range(double low, double high, double step) {
    vector<double> range;
    for (double value = low; value <= high; value += step) {
        range.push_back(value);
    }
    return range;
}

// Worker function to run the benchmark
void benchmark_worker(int pc, double c, double m, int steps, function<double(double)> fun, double local_extr_v, 
                      vector<tuple<int, double, double, double>>& local_results, mutex& results_mutex) {
                           
    

    Population<1, 10, 5> pop = {};
    pop.fill(pc);
    pop.setFunction(fun);
    pop.setPCross(c);
    pop.setPMut(m);
    pop.doSteps(steps);
    

    double error = local_extr_v - fun(pop.get_best_point());
    error*=error;


    {
        lock_guard<mutex> guard(results_mutex);
        //cout<<"thread "<<pc<<c<<m<<" ended"<<endl;
        local_results.push_back({pc, c, m, error});
    }
}


vector<tuple<int, double, double, double>> gen_benchmark(int population_min, int population_max,int population_step, double low_c, double high_c, double step_c, 
                   double low_m, double high_m, double step_m, int steps, function<double(double)> fun, double fun_extr) {


    double local_extr;
    local_extr = fun_extr;
    double local_extr_v = fun(local_extr);


    auto start_time = chrono::high_resolution_clock::now();

    vector<tuple<int, double, double, double>> results; 
    vector<thread> threads; 
    mutex results_mutex; 

    for( int pc = population_min; pc<=population_max; pc+=population_step){
        for(double c = low_c;c<=high_c;c += step_c){
            for(double m = low_m;m<=high_m;m += step_m){
               threads.emplace_back(benchmark_worker,pc,c,m,steps,fun,local_extr_v,ref(results),ref(results_mutex));
            }
        }
    }

     for (auto& thread : threads) {
        thread.join();
    }
    auto end_time = chrono::high_resolution_clock::now(); 
    chrono::duration<double> duration = end_time - start_time;

    cout << "Benchmark completed in " << duration.count() << " seconds." << endl;
    return results;
}
long estimateTotalIterations(int population_min, int population_max, int population_step,
                            double low_c, double high_c, double step_c,
                            double low_m, double high_m, double step_m) {
    // Calculate the number of iterations for each parameter
    long population_iterations = (population_max - population_min) / population_step + 1;

    long crossover_iterations = static_cast<long>((high_c - low_c) / step_c) + 1;
    long mutation_iterations = static_cast<long>((high_m - low_m) / step_m) + 1;

    // Total iterations is the product of all iterations
    return population_iterations * crossover_iterations * mutation_iterations;

}
void writeDataToFile(const string& filename, const vector<tuple<int, double, double, double>>& data) {
    ofstream ofs(filename);
    if (!ofs) {
        cerr << "Error opening file for writing: " << filename << endl;
        return;
    }

    for (const auto& entry : data) {
        int id = get<0>(entry);
        double x = get<1>(entry);
        double y = get<2>(entry);
        double color = get<3>(entry);
        ofs << x << " " << y << " " << color << endl;
    }
    ofs.close();
}
void plotDataWithGnuplot(const string& filename) {
    Gnuplot plt{};

    plt.sendcommand("set palette defined (0 'blue', 1 'green', 2 'yellow', 3 'red'); "
                                   "set cblabel 'Color Value'; "
                                   "set xlabel 'X Axis'; "
                                   "set ylabel 'Y Axis'; "
                                   "set zlabel 'Z Axis'; "
                                   "splot '" + filename + "' using 1:2:3 with points palette pointsize 2 title '3D Plot with Color'");


}
int main(){
    const double time = double(135)/double(51272*20);
    cout<<estimateTotalIterations(16, 1000, 4, 0.2, 0.8, 0.05, 0.005, 0.4, 0.05)<<endl;
    cout<<"Estimated time:"<<estimateTotalIterations(16, 1000, 4, 0.2, 0.8, 0.05, 0.005, 0.4, 0.05)*time*20<<"'s"<<endl;
    auto results = gen_benchmark(16, 1000, 4, 0.2, 0.8, 0.05, 0.005, 0.4, 0.05, 20, 
                                  [](double x) { return log(x)*cos(3*x-15); }, 9.19424);
   
    
    sort(results.begin(),results.end(),[](tuple<int, double, double, double> a,tuple<int, double, double, double> b){return get<3>(a)<get<3>(b);});
    int cnt =0;
    for (const auto& result : results) {
        cout << "Population: " << get<0>(result)
             << ", Crossover: " << get<1>(result)
             << ", Mutation: " << get<2>(result)
             << ", Error: " << get<3>(result) << endl;
        if(cnt++>=10){
            break;
        }
    }
    writeDataToFile("results.txt",results);
    plotDataWithGnuplot("results.txt");
 
    return 0;
}