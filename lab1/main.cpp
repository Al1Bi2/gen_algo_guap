#include <bitset>
#include <random>
#include <cmath>
#include <iostream>
#include <functional>
#include "../libs/gplot++.h"

using namespace std;
int seed = 42;
random_device rnd;
mt19937 gen(rnd());

long random(long min, long max){
std::uniform_int_distribution<long> distrib(min, max);
return distrib(gen);
}
long randomd(double min, double max){
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
    constexpr Population(){}
    static constexpr long n(){return static_cast<long>((max-min)*powl(10,dad));}
    static constexpr long lenght(){return static_cast<long>(log2(n()))+1;}

    
    void reproduction(){
        vector<Genome> mid_population  ={};
        vector<double> y  ={};
        double avg = 0;
        for(auto gene: population){
            double fy = function(static_cast<double>(gene.getGenome()));
            y.push_back(fy);
            avg+=fy;
        }
        avg/=population.size();
        for(int i = 0; i < population.size(); i++){
            float descendants = std::floor(y[i]/avg);
            for(int j = 0; j < descendants;j++){
                mid_population.push_back(population[i]);
            }

        }
        
    }
    void crossingover(double chance){
        vector<Genome> mid_population  ={};
        
        vector<Genome> a_half  ={};
        vector<Genome> b_half  ={};
        long a,b;
        while(mid_population.size!=0){
            do{
                a = random(0,lenght());
                b = random(0,lenght());
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
            double rand_v  = random(0.0d, 1.0d);
            if(rand_v <chance){
                swap<lenght()>(a_half[i].gene,b_half[i].gene,idx);
            }     
            
        }
        mid_population.emplace_back(a_half);
        mid_population.emplace_back(b_half);
       
    }

    void mutation(double chance){
        std:vector<Genome> mid_population  ={};
        for(auto g: mid_population){
            double rand_v = random(0.0d, 1.0d);
            if(rand_v < chance){
                long idx =  random(0,lenght());
                g.gene.flip(idx);
            }
        }

    }


    std::function<double(double)> function;

    class  Genome{
    public:
        bitset<lenght()> gene;
    public:
        void setGenome(unsigned long value){
            cout<<value<<endl;
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
        
        return random(0,n());
    }
    double pos_to_real(long pos){
        return pos*max/static_cast<double>(n())+min;
    }
    void fill( int count){
        for(int i = 0;i < count; i++){
            Genome g;
            g.setGenome(generate());
            population.push_back(g);
        }
    }
    void print(){
        for(auto p: population){
            cout<<p.getGenome()<<" ";
        }
    }
    void draw(){
        
    }
    
};
int main(){
    std::vector<double> x,y;  // No problem to use a vector of ints

    Population<0,10,3> p ={};
    p.fill(200);
    p.function = [&p](double x){return log(x)*cos(3*x-15);};
    //auto function = [&p](double x)->double {return (1.85-x)*cos(3.5*x-0.5);};
Gnuplot plt{};
for(double i = 0; i<10;i+=0.001){
    x.push_back(i);
    y.push_back(p.function(i)); 
    

}
std::vector<double> px,py;
for(auto pop: p.population){
    px.push_back(p.pos_to_real(pop.gene.to_ulong()));
    py.push_back(p.function(px.back()));
}

// You can provide a label and a linestyle
plt.plot(x, y, "Dataset #1", Gnuplot::LineStyle::LINES);
plt.plot(px, py, "Dataset #2", Gnuplot::LineStyle::POINTS);
// Now produce the plot
plt.show();

    
    //p.print();
    return 0;
}