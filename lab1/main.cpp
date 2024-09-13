#include <bitset>
#include <random>
#include <cmath>
#include <iostream>
#include <functional>
using namespace std;
int seed = 42;
random_device rnd;
mt19937 gen(rnd());

long random(long min, long max){
std::uniform_int_distribution<long> distrib(min, max);
return distrib(gen);
}

template<int min, int max,int dad>
class Population{
public:
    constexpr Population(){}

    
    void reproduction(){
        std:vector<long>  = {};
        std:vector<Genome<lenght()>> mid_population  ={};
        doule avg = 0;
        for(auto gene: population){
            fy = function(gene.getGenome()
            y.push_back(fy));
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
    void crossingover(){
        std:vector<Genome<lenght()>> a_half  ={};
        std:vector<Genome<lenght()>> b_half  ={};
        

       
    }
    std::function<long(long)> function;
    template<size_t bits>
    class  Genome{
    private:
        bitset<bits> gene;
    public:
        void setGenome(unsigned long value){
            gene = bitset<bits>(value);
        }
        long getGenome(){
            return gene.to_ulong();
        }
    };
public:
    static constexpr long n(){return static_cast<long>((max-min)*powl(10,dad));}
    static constexpr long lenght(){return static_cast<long>(log2(n()))+1;}
    const double max_value = (max-min)*powl(10,dad);
    vector<Genome<lenght()>>  population = {};
    long generate(){
        return random(0,n());
    }
    void fill(){
        for(int i = 0;i < n()/200; i++){
            Population<min,max,dad>::Genome<lenght()> g;
            g.setGenome(generate());
            population.push_back(g);
        }
    }
    void print(){
        for(auto p: population){
            cout<<p.getGenome()<<" ";
        }
    }
    std::vector<std::vector<Genome<lenght()>>> split(){
        std::vector<Genome<lenght()>>  p =  population;

        
        std::vector<Genome<lenght()> a={};
        
        std::vector<Genome<lenght()> b={};
        for(int i = 0; i < population.size();i++){
            int rand_pos =  
        }


    }
};
int main(){
    
    Population<0,10,3> p ={};
    p.fill();
    p.print();
    return 0;
}