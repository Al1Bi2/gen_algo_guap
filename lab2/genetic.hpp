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
        enum  EXTREMUM{
            MAX = 1,
            MIN = -1
        };
        class roulette{
        public:
            EXTREMUM dir;
            roulette(EXTREMUM dir = EXTREMUM::MAX):dir(dir){};

            std::pair<size_t,size_t> doit(const std::vector<double>& fitness){
                double sum = 0;
                double max = *std::max_element(fitness.begin(),fitness.end());
                double min = *std::min_element(fitness.begin(),fitness.end());
                for (double v : fitness) {
                    sum += dir == EXTREMUM::MAX ? v - min : max - v;
                }
                double r = random_double(0.0, sum);
                double current_sum = 0;
                size_t idx1 = 0, idx2 = 0;
                for (size_t i = 0; i < fitness.size(); i++) {
                    current_sum += dir == EXTREMUM::MAX ? fitness[i] - min : max - fitness[i];
                    if (r < current_sum) {
                        idx1 = i;
                        break;
                    }
                }
                r = random_double(0.0, sum);
                current_sum = 0;
                for (size_t i = 0; i < fitness.size(); i++) {
                    current_sum += dir == EXTREMUM::MAX ? fitness[i] - min : max - fitness[i];
                    if (r < current_sum) {
                        idx2 = i;
                        break;
                    }
                    
                }
                
                return {idx1,idx2};
            }
        };
        class tournament{
        public:
            size_t tournament_size;
            EXTREMUM dir;
            tournament(size_t ts = 3,EXTREMUM dir = EXTREMUM::MAX):tournament_size(ts),dir(dir){};
            
            std::pair<size_t,size_t> doit(const std::vector<double>& fitness){
                std::vector<size_t> candidates;
                size_t size = fitness.size();
                for (size_t i = 0; i < tournament_size; i++) {
                    candidates.push_back(random_long(0, size - 1));
                }
                auto lambda  = [&fitness, this](size_t a, size_t b) {
                    return this->dir == EXTREMUM::MIN ? fitness[a] > fitness[b] : fitness[a] < fitness[b];
                };
                
                size_t parent1_idx = *std::max_element(candidates.begin(), candidates.end(),
                    lambda);
                candidates.erase(std::remove(candidates.begin(), candidates.end(), parent1_idx), candidates.end());
                size_t parent2_idx = *std::max_element(candidates.begin(), candidates.end(),
                    lambda);

                return { parent1_idx, parent2_idx };
            }

        };
    }
    namespace crossover{
        namespace RGA{

            class BLX{
            public:
                double alpha;
                BLX(double alpha = 0.5):alpha(alpha){};

                void doit(const std::vector<double>& parent1, const std::vector<double>& parent2, std::vector<double>& child1, std::vector<double>& child2){
                    //std::vector<double> c1(a.size()), c2(a.size());
                    int n = parent1.size();
                    child1.resize(n);
                    child2.resize(n);
                    for(size_t i = 0; i < parent1.size(); i++){
                        double cmin = std::min(parent1[i],parent2[i]);
                        double cmax = std::max(parent1[i],parent2[i]);
                        double di = cmax - cmin;
                        child1[i] = cmin + random_double(-alpha*di, (1+alpha)*di);
                        child2[i] = cmin + random_double(-alpha*di, (1+alpha)*di);
                    }
                    //return {c1, c2};
                }

            };
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
                            beta = pow(2 * u/* + (1 - 2 * u) * pow(1 - u, eta)*/, 1.0 / (eta + 1.0));
                        } else {
                            beta = pow(1 / (2 * (1 - u) /*+ 2 * (u - 0.5) * pow(1 - u, eta)*/), 1.0 / (eta + 1.0));
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

            non_uniform(double min_value = 0.0, double max_value = 1.0, double eta = 5.0)
                :  min_value(min_value), max_value(max_value), eta(eta) {}

            
            void doit(std::vector<double>& genotype, size_t generation, size_t max_generations) {
                int idx = random_long(0, genotype.size() - 1);
                int dir = random_long(0, 1);
                double diff;
                
                double r = random_double(0.0, 1.0);
                double delta = (1-pow(r, pow(1-generation/max_generations, eta)));
                if(dir){
                    diff = -(genotype[idx] - min_value);
                }else{
                    diff = max_value - genotype[idx];
                }
                genotype[idx] += delta * diff;

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
        
        population(): popul(){};
        population(size_t size): popul(size){};
        std::vector<std::pair<T,double>> get(){
            return popul;
        }
        size_t size(){
            return popul.size();
        }
        T& at(size_t idx){
            return popul[idx].first;
        }
        double& fit_at(size_t idx){
            return popul[idx].second;
        }
        void push_back(T elem){
            popul.push_back({elem,0.0});

        }
        void append_range(population& other){
            auto other_content  =other.get();
            int old_size = this->size();
            this->resize(old_size+other.size());
            for(int i = old_size; i< this->size();i++){
                popul[i] = other_content[i-old_size];
            }
            //popul.append_range(other.get());
        }
        void fill(std::vector<double> min,std::vector<double> max){
            for(int i = 0;i<popul.size();i++){
                popul[i].first.rnd(min,max);
                popul[i].second = 0;
            }
        }
        void resize(size_t size){
            popul.resize(size);
        }
        void clear(){
            popul.clear();
        }
        std::pair<T,double> best(gen::reproduction::EXTREMUM dir = gen::reproduction::EXTREMUM::MAX){
            size_t best_idx  = -1;
            double best_fitness=dir == gen::reproduction::EXTREMUM::MAX ? std::numeric_limits<double>::min() : std::numeric_limits<double>::max();
            for(int i = 0;i<this->size();i++){
                bool is_better = dir == gen::reproduction::EXTREMUM::MAX ? popul[i].second > best_fitness : popul[i].second < best_fitness;
                if(is_better){
                    best_idx = i;
                    best_fitness = popul[i].second;
                }
            }
            return popul[best_idx];
        }
        void print(){
            for(int i = 0;i<this->size();i++){
               std::cout<<" ("<<popul[i].first.get()[0]<<","<<popul[i].first.get()[1]<<"):"<<popul[i].second<<std::endl;
            }
        }
        void sort(gen::reproduction::EXTREMUM dir = gen::reproduction::EXTREMUM::MAX){
            auto comp = [dir](std::pair<T,double> a,std::pair<T,double> b){return dir== gen::reproduction::EXTREMUM::MAX ? a.second>b.second:a.second<b.second;};
            std::sort(popul.begin(),popul.end(), comp);
        }
        std::vector<double> get_fit(){
            std::vector<double> res;
            for(auto& elem: popul){
                res.push_back(elem.second);
            }
            return res;
        }
        void clamp(std::vector<double> min, std::vector<double> max){
            for(auto& elem: popul){
                for(int i = 0;i<elem.first.get().size();i++){
                    elem.first.get()[i] = std::clamp(elem.first.get()[i],min[i],max[i]);
                }
            }
        }
    public:
        std::vector<std::pair<T,double>> popul;


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
        gen::reproduction::EXTREMUM dir;

        GA<GeneType,ReproductionPolicy,MutationPolicy,CrossoverPolicy>(GeneType gt, ReproductionPolicy rp, MutationPolicy mp, CrossoverPolicy cp, gen::reproduction::EXTREMUM dir = gen::reproduction::EXTREMUM::MAX){
            population_size = 20;
            population = gen::population<GeneType>(population_size);
            new_population = gen::population<GeneType>(population_size);
            this->dir = dir;
            reproduction_p =rp;
            reproduction_p.dir = dir;
            mutation_p = mp;
            crossover_p= cp;
        }
        void set_population_size(int size){
            this->population_size=size;
            population.resize(population_size);
            new_population.resize(population_size);
        }
        void set_algo(std::function<double(std::vector<double>)> algo){
                this->algo = algo;           
        }
        void fill(std::vector<double> min, std::vector<double> max){
            population.fill(min,max);
            this->min_borders = min;
            this->max_borders = max;         
        }
       
        void calculate_fitness(){
            if (!algo) {
                throw std::runtime_error("Fitness algorithm (algo) not set.");
            }
            for(int i =0; i< population.size(); i++){
                population.fit_at(i) = algo(population.at(i).get());
            }
        }


        std::pair<GeneType&,GeneType&> reproduct(){
            std::pair<size_t, size_t> parents_idx = reproduction_p.doit(population.get_fit());
            return { population.at(parents_idx.first), population.at(parents_idx.second) };
        }
        void crossover(){
            int n = population.size();
             std::vector<std::vector<double>> v;
            for(int i = 0; i< n/2; i++){
                std::vector<double> child1, child2;
                const std::pair<GeneType&,GeneType&> parents = reproduct();
                v.push_back(parents.first.get());
                 v.push_back(parents.first.get());
                crossover_p.doit(parents.first.get(),parents.second.get(),child1,child2);
                new_population.at(i*2) = GeneType(child1);
                new_population.at(i*2+1)=GeneType(child2); 
            }
        }       
        
        void mutate(){

            for(int i = 0; i< population.size();i++){
                mutation_p.doit(new_population.at(i).get(),current_step,max_step);
            }
        }
        void reduce(){
            calculate_fitness();
            population.sort(dir);
            population.resize(10);
            population.append_range(new_population);
            calculate_fitness();
            population.sort(dir);
            population.clamp(min_borders,max_borders);

            
            //population.print();
            population.resize(population_size);
            
        }

        void doit(){
            calculate_fitness();
            crossover();
     
            mutate();
            
            reduce();

            
            current_step++;
        }
        void doit(size_t n){
            for(size_t i = 0; i< n;i++){
                doit();
            }
        }
        std::vector<std::pair<GeneType,double>> get_population(){
            std::vector<std::pair<GeneType,double>> ret;
            for(int i = 0; i< population.size();i++){
                ret.push_back({population.at(i),population.fit_at(i)});
            }
            return ret;
        }
    };}