#pragma once

#include <vector>
#include <bitset>
#include <algorithm>
#include <execution>
#include <functional>
#include <random>
#include <utility>
namespace gen{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    double random_double(double a, double b) {
        std::uniform_real_distribution<> dis(a, b);
        return dis(gen);
    }

    
long random_long(long min, long max){
    std::uniform_int_distribution<long> distrib(min, max);
    return distrib(gen);
}
    
    namespace reproduction{
        class roulette{

        };
        class tournament{
        public:
            size_t tournament_size;
            tournament(size_t ts = 3):tournament_size(ts){};

            std::pair<size_t,size_t> doit(const std::vector<double>& fitness){
                std::vector<size_t> candidates;
                size_t size = fitness.size();
                for (size_t i = 0; i < tournament_size; i++) {
                    candidates.push_back(random_long(0, size - 1));
                }

                size_t parent1_idx = *std::max_element(candidates.begin(), candidates.end(),
                    [&fitness](size_t a, size_t b) {
                        return fitness[a] < fitness[b];
                    });
                candidates.erase(std::remove(candidates.begin(), candidates.end(), parent1_idx), candidates.end());
                size_t parent2_idx = *std::max_element(candidates.begin(), candidates.end(),
                    [&fitness](size_t a, size_t b) {
                        return fitness[a] < fitness[b];
                    });

                return { parent1_idx, parent2_idx };
            }

        };
    }
    namespace crossover{
        namespace RGA{

            class SBX{
            public:
                double eta;
                SBX(double eta=2):eta(eta){};

                void doit(const std::vector<double>& parent1,const std::vector<double>& parent2,
                                std::vector<double>& child1, std::vector<double>& child2){
                    
                    int n = parent1.size();
                    child1.resize(n);
                    child2.resize(n);
                    for(int i = 0; i< n; i++){
                        double u = random_double(0.0,1.0);
                        double beta;
        
                        if (u <= 0.5) {
                            beta = pow(2 * u + (1 - 2 * u) * pow(1 - u, eta), 1.0 / (eta + 1.0));
                        } else {
                            beta = pow(1 / (2 * (1 - u) + 2 * (u - 0.5) * pow(1 - u, eta)), 1.0 / (eta + 1.0));
                        }
                        child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i]);
                        child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i]);
                    }
                    
                }
                
            };

            
        }
    }
    namespace mutation{
        class non_uniform{
        public:
            double mutation_rate; 
            double min_value;    
            double max_value;     
            double eta;         

            non_uniform(double mutation_rate = 0.01, double min_value = 0.0, double max_value = 1.0, double eta = 20.0)
                : mutation_rate(mutation_rate), min_value(min_value), max_value(max_value), eta(eta) {}

            
            void doit(std::vector<double>& genotype, size_t generation, size_t max_generations) {
                for (size_t i = 0; i < genotype.size(); ++i) {
                    if (random_double(0.0, 1.0) < mutation_rate) {
                        double delta = (max_value - min_value) * (1.0 - generation / static_cast<double>(max_generations));
                        double random_value = random_double(-delta, delta);
                        
                        // Применяем мутацию с уменьшением интенсивности
                        genotype[i] += random_value;
                        
                        // Ограничиваем новое значение гена диапазоном [min_value, max_value]
                        if (genotype[i] < min_value) genotype[i] = min_value;
                        if (genotype[i] > max_value) genotype[i] = max_value;
                    }
                }
            }
        };
    }
    namespace genome{
        class BGA{
            std::vector<bool> gene;
        };
        template<size_t size>
        class RGA{
        public:
            std::vector<double> genotype;
            RGA():genotype(size){};
            RGA(const std::vector<double>& genes ):genotype(genes){};
            //RGA(size_t size):genotype(size){};
            std::vector<double>& get(){
                return genotype;
            }
            void rnd(const std::vector<double>& min, const std::vector<double>& max){
                for(int i = 0; i< size; i++){
                    genotype[i] = random_double(min[i],max[i]);
                }
            }
        };
    }

    template <typename T>
    class population{
    public:
        
        population(): popul(),fitness(){};
        population(size_t size): popul(size),fitness(size){};
        size_t size(){
            return popul.size();
        }
        T& at(size_t idx){
            return popul[idx];
        }
        double& fit_at(size_t idx){
            return fitness[idx];
        }
        void push_back(T elem){
            popul.push_back(elem);
            fitness.push_back(0.0);
        }
        void fill(std::vector<double> min,std::vector<double> max){
            for(int i = 0;i<popul.size();i++){
                popul[i].rnd(min,max);
                fitness[i] = 0;
            }
        }
        void resize(size_t size){
            popul.resize(size);
            fitness.resize(size);
        }
        std::pair<T,double> best(){
            size_t best_idx  = -1;
            double best_fitness=std::numeric_limits<double>::min();
            for(int i = 0;i<this->size();i++){
                if(fitness[i]> best_fitness){
                    best_fitness = fitness[i];
                    best_idx=i;
                }
            }
            return {popul[best_idx],fitness[best_idx]};
        }
        void print(){
            for(int i = 0;i<this->size();i++){
                std::cout<<popul[i].get()[0]<<":"<<fitness[i]<<std::endl;
            }
        }
    public:
        std::vector<T> popul;
        std::vector<double> fitness;

    };

    template <typename T>
    class algo{
    public:
        algo(){};
        population<T> p;
        


    };
    template<typename GeneType,typename ReproductionPolicy, typename MutationPolicy, typename CrossoverPolicy>
    class GA{
    public:
        gen::population<GeneType> population;
        gen::population<GeneType> new_population;
        ReproductionPolicy reproduction_p;
        MutationPolicy mutation_p;
        CrossoverPolicy crossover_p;
        size_t population_size;
        size_t current_step=0;
        size_t max_step=100;
        std::vector<double> min_borders;
        std::vector<double> max_borders;
        std::function<double(const std::vector<double>)> algo;

        GA<GeneType,ReproductionPolicy,MutationPolicy,CrossoverPolicy>(GeneType gt, ReproductionPolicy rp, MutationPolicy mp, CrossoverPolicy cp){
            population_size = 20;
            population = gen::population<GeneType>(population_size);
            new_population = gen::population<GeneType>(population_size);
            reproduction_p =rp;
            mutation_p = mp;
            crossover_p= cp;
            
        }
        void set_popuation_size(int size){
            this->population_size=size;
            population.resize(population_size);
            new_population.resize(population_size);
        }
        void fill(std::vector<double> min, std::vector<double> max){
            population.fill(min,max);
            this->min_borders = min;
            this->max_borders = max;
            
        }
        void set_algo(std::function<double(std::vector<double>)> algo){
            this->algo = algo;
        }
        void calculate_fitness(){
            for(int i =0; i< 20; i++){
                population.fit_at(i) = algo(population.at(i).get());
            }
        }

        std::pair<GeneType&,GeneType&> reproduct(){
            std::pair<size_t, size_t> parents_idx = reproduction_p.doit(population.fitness);
            return { population.at(parents_idx.first), population.at(parents_idx.second) };
        }
        void crossover(){
            int n = population.size();
            for(int i = 0; i< n/2; i++){
                std::vector<double> child1, child2;
                const std::pair<GeneType&,GeneType&> parents = reproduct();

                crossover_p.doit(parents.first.get(),parents.second.get(),child1,child2);
                new_population.at(i) = GeneType(child1);
                new_population.at(i+1)=GeneType(child2); 
            }
        }
        void mutate(){
            for(int i = 0; i< population.size();i++){
                mutation_p.doit(population.at(i).get(),current_step,max_step);
            }
        }
        void doit(){
            calculate_fitness();
            crossover();
            new_population.print();
            std::cout<<"+++++++++++++"<<std::endl;
            mutate();
            new_population.print();
            std::cout<<"+++++++++++++"<<std::endl;
            population = new_population;
            calculate_fitness();
            population.print();
            std::cout<<"+++++++++++++"<<std::endl;
            current_step++;
        }
        void doit(size_t n){
            for(size_t i = 0; i< n;i++){
                doit();
            }
        }
    };
}