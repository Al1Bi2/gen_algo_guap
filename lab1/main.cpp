#include <bitset>
#include <random>
#include <cmath>
#include <iostream>
#include <functional>
#include <iterator>
#include <vector>
#include "../libs/gplot++.h"

using namespace std;
int seed = 42;
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
template<size_t len>
void swap(std::bitset<len>& a,std::bitset<len>& b, size_t end_idx){

    for(size_t i = 0; i < end_idx; i++){
        bool tmp = a[i];
        a[i] = b[i];
        b[i] = tmp;  
    }
}

template<int min, int max,int dad>
class Population{
public:
    Gnuplot plt{};
    class Genome;
    constexpr Population(){}
    static constexpr long lenght(){return static_cast<long>(log2((max-min)*pow(10,dad)))+1;};
    static constexpr long n(){return static_cast<long>(powl(2,lenght()));}
    OptimizationType optimization_type = OptimizationType::MAXIMIZATION;
    vector<Genome> mid_population  ={};
    
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

    void crossingover(double chance= 0.7d){

        
        vector<Genome> a_half  ={};
        vector<Genome> b_half  ={};
        long a,b;
        while(mid_population.size()!=0){
            do{
                a = random(0,mid_population.size() -1);
                b = random(0,mid_population.size()-1);
            }while(a==b);
            auto a_it = std::next(mid_population.begin(), a);
            auto b_it = std::next(mid_population.begin(), b);
            a_half.push_back(*a_it);
            b_half.push_back(*b_it);
            mid_population.erase(a_it);
            mid_population.erase(b_it);
        }
        for(int i = 0; i < a_half.size();i++){
            long idx =  random(1,lenght()-1);
            double rand_v  = randomd(0.0d, 1.0d);
            if(rand_v <chance){
                swap<lenght()>(a_half[i].gene,b_half[i].gene,idx);
            }     
            
        }
        mid_population.insert(mid_population.end(),a_half.begin(),a_half.end());
        mid_population.insert(mid_population.end(),b_half.begin(),b_half.end());
       
    }

    void mutation(double chance = 0.001d){

        for(auto g: mid_population){
            double rand_v = randomd(0.0d, 1.0d);
            if(rand_v < chance){
                long idx =  random(0,lenght()-1);
                g.gene.flip(idx);
            }
        }
        //printm();
        population.clear();
        for(auto pop: mid_population){
            population.push_back(pop);
        }
        mid_population.clear();

    }


    std::function<double(double)> function;

    class  Genome{
    public:
        bitset<lenght()> gene;
    public:
        void setGenome(unsigned long value){
            gene = bitset<lenght()>(value);
            
        }
        long getGenome(){
            return gene.to_ulong();
        }
    };
public:
    
    const double max_value = (max-min)*powl(10,dad);
    vector<Genome>  population = {};
    long generate(){
        
        return random(0,n()-1);
    }
    double pos_to_real(long pos){
        return pos*(max-min)/static_cast<double>(n()-1)+min;
    }
    void fill( int count){
        for(int i = 0;i < count; i++){
            Genome g;
            g.setGenome(generate());
            population.push_back(g);
        }
    }
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
    void draw(){
        
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

        // You can provide a label and a linestyle
        plt.plot(x, y, "Dataset #1", Gnuplot::LineStyle::LINES);
        plt.plot(px, py, "Dataset #2", Gnuplot::LineStyle::POINTS);
        // Now produce the plot
        plt.show();
    
                
    }
    void step(){
        reproduction();
        crossingover();
        mutation();
        //printp();
    }
    
};
int main(){
    int n=0;
    Population<-10,10,3> p ={};
    p.fill(100);
    //p.function = [&p](double x){return log(x)*cos(3*x-15);};
    //p.function = [&p](double x){return x*x;};
    p.function = [&p](double x)->double {return (1.85-x)*cos(3.5*x-0.5);};
 
    p.draw();

    p.plt.reset();
    cin>>n;
    while(n!=1){
  
        p.step();

        p.draw();

        cin>>n;
    }

        
        //p.print();

    return 0;
}