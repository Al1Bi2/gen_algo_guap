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
#include <array>
#include <atomic>
#include "../libs/gplot++.h"

using namespace std;
atomic<int> count_joined(0);

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
    double get_genome(int i){
        return pos_to_real(population[i].getGenome());
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
                      vector<tuple<int, double, double, double,double>>& local_results, mutex& results_mutex) {
                           
    

    Population<1, 10, 5> pop = {};
    pop.fill(pc);
    pop.setFunction(fun);
    pop.setPCross(c);
    pop.setPMut(m);
    int n = 0;
    double last_error = 123456789;
    int close_points = 0;
    const double close_dist = (10-1)/double(100);
    double best_point = pop.get_best_point();
    auto start_time = chrono::high_resolution_clock::now();
    while(n++<steps){
        pop.step();
        for(int i = 0;i<pc;i++){
            double error = pop.get_genome(i) - best_point;
            if(fabs(error) > close_dist*2) {
                continue;
            }
            if(abs(error)<close_dist){
                close_points++;
            }
            if(close_points>=(pc*0.9)){
            break;
            }
        } 

        if(close_points>=double(pc*0.9)+1){
            break;
        } 
    
    }
    auto end_time = chrono::high_resolution_clock::now(); 
    chrono::duration<double> duration = end_time - start_time;
    
    double error = abs(fun(pop.get_best_point())-local_extr_v);
    {
        lock_guard<mutex> guard(results_mutex);
        //cout<<"thread "<<pc<<c<<m<<" ended"<<endl;
        local_results.push_back({pc, c, m,duration.count(),error});
        
    }
}
void monitor_progress(int total_threads) {
    int percent_size = total_threads / 100;

    
    while (count_joined.load() < total_threads*0.8) {
        cout << "Progress: " << count_joined.load() / percent_size << "%" << endl;
        this_thread::sleep_for(chrono::milliseconds(500));  // Check every 500ms
        int r = count_joined.load();
        int r2=2+r;
    }
}

long estimate_iterations(int population_min, int population_max, int population_step,
                            double low_c, double high_c, double step_c,
                            double low_m, double high_m, double step_m) {
    // Calculate the number of iterations for each parameter
    long population_iterations = (population_max - population_min) / population_step + 1;

    long crossover_iterations = static_cast<long>((high_c - low_c) / step_c) + 1;
    long mutation_iterations = static_cast<long>((high_m - low_m) / step_m) + 1;

    // Total iterations is the product of all iterations
    return population_iterations * crossover_iterations * mutation_iterations;

}
vector<tuple<int, double, double, double,double>> gen_benchmark(int population_min, int population_max,int population_step, double low_c, double high_c, double step_c, 
                   double low_m, double high_m, double step_m, int steps, function<double(double)> fun, double fun_extr) {


    double local_extr;
    local_extr = fun_extr;
    double local_extr_v = fun(local_extr);


    auto start_time = chrono::high_resolution_clock::now();

    vector<tuple<int, double, double, double, double>> results; 
    vector<thread> threads; 
    mutex results_mutex; 
    thread progress_thread(monitor_progress, estimate_iterations(population_min, population_max, population_step, low_c, high_c, step_c, low_m, high_m, step_m));
    for( int pc = population_min; pc<=population_max; pc+=population_step){
        for(double c = low_c;c<=high_c;c += step_c){
            for(double m = low_m;m<=high_m;m += step_m){
               threads.emplace_back(benchmark_worker,pc,c,m,steps,fun,local_extr_v,ref(results),ref(results_mutex));
               count_joined++;
            }
        }
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    progress_thread.join();
    auto end_time = chrono::high_resolution_clock::now(); 
    chrono::duration<double> duration = end_time - start_time;

    cout << "Benchmark completed in " << duration.count() << " seconds." << endl;
    return results;
}



template <typename T>
void write_to_file(const string& filename, const vector<T>& data) {
    ofstream ofs(filename);
    if (!ofs) {
        cerr << "Error opening file for writing: " << filename << endl;
        return;
    }
    for (const auto& entry : data) {
            std::apply(
                [&ofs](auto&&... args) {
                    ((ofs << args << " "), ...);
                    ofs << endl;
                },
                entry);
        }
        ofs.close();
    }

void plot_gnu(const string& filename,const vector<int>& columns, const array<string,5>& titles, bool is_log=0) {
    Gnuplot plt{};
    string labels="";
    string log="";
    string plot_type = "";
    string columns_str = "";
    if(columns.size()==2){
        labels = "set xlabel '"+titles[columns[0]]+"'; set ylabel '"+titles[columns[1]]+"'; ";
        if(is_log){
            log = "set logscale y; ";
        }
        plot_type= "plot";
        columns_str=to_string(columns[0]+1)+":"+to_string(columns[1]+1)+":"+to_string(columns[1]+1);
    }else if(columns.size()==3){
        labels = "set xlabel '"+titles[columns[0]]+"'; set ylabel '"+titles[columns[1]]+"'; set zlabel '"+titles[columns[2]]+"'; ";
        if(is_log){
            log = "set logscale z; ";
        }
        plot_type= "splot";
        columns_str=to_string(columns[0]+1)+":"+to_string(columns[1]+1)+":"+to_string(columns[2]+1);
    }else{
        cout<<"Wrong number of columns"<<endl;
        return;
    }
    string result = "set palette defined (0 'blue', 1.00E-17 'green', 1.00E-10 'yellow',1.00E-8 'red'); "
                                   "set cblabel 'Error Value'; "
                                   +labels+" "
                                   +log
                                    +" set style fill solid 0.5 noborder; " 
                                    +plot_type+" '" + "results.txt" + "' using "+ columns_str +  " with points  pt 1 ps 0.8 palette title '" + "GRAPH" + "'";
    plt.sendcommand(result);


}
int main(){
    const array<string,5> columns_title = {"Population", "Crossingover", "Mutation","Time" ,"Error"};
    const string filename1 = "3d_data1.txt";
    const string filename2 = "3d_data2.txt";
    const string filename3 = "3d_data3.txt";
    int n=-1;
    while(n!=9){
        cout<<"Choose option:"<<endl;
        cout<<"\t1)Recalculate benchmark and save to file"<<endl;
        cout<<"\t2)Plot files"<<endl;
        cout<<"\t3)Load individual simulation"<<endl;
        cout<<"\t9)Exit"<<endl;
        cin>>n;
        switch(n){
            case 1:
            {
                const double time = double(135)/double(51272*20);
                int steps = 100;
                int min_pop=2;
                int max_pop = 500;
                int step_pop = 2;
                double min_cop=0.1;
                double max_cop = 0.9;
                double step_cop = 0.05;
                double min_mp=0.0001;
                double max_mp = 0.4;
                double step_mp = 0.01;
                auto fun = [](double x) { return log(x)*cos(3*x-15); };
                double local_extr =9.19424;
                long est_iterations = estimate_iterations(min_pop, max_pop, step_pop, min_cop, max_cop, step_cop, min_mp, max_mp, step_mp);
                cout<<"Estimated iterations:"<<est_iterations<<endl;
                cout<<"Estimated time:"<<est_iterations*time*steps<<"'s"<<endl;
                auto results = gen_benchmark(min_pop, max_pop, step_pop, min_cop, max_cop, step_cop, min_mp, max_mp, step_mp, steps,fun,local_extr);
                sort(results.begin(), results.end(),[](auto a,auto b){return get<3>(a)<get<3>(b);});
                int cnt = 0;
                cout<<"Top 20 results:"<<endl;
                for (const auto& result : results) {
                    cout << "Population: " << get<0>(result)
                        << ", Crossover: " << get<1>(result)
                        << ", Mutation: " << get<2>(result)
                        << ", Error: " << get<3>(result) << endl;
                    if(cnt++>=20){
                        break;
                    }
                }
                write_to_file<tuple<int, double, double, double,double>>("results.txt", results);
                vector<tuple<double, double, double>> data1, data2, data3;
                for (const auto& entry : results) {
                    data1.emplace_back(get<0>(entry), get<1>(entry), get<3>(entry)); 
                    data2.emplace_back(get<1>(entry), get<2>(entry), get<3>(entry)); 
                    data3.emplace_back(get<2>(entry), get<0>(entry), get<3>(entry)); 
                    }
                
                write_to_file<tuple<double, double, double>>("3d_data1.txt", data1);
                write_to_file<tuple<double, double, double>>("3d_data2.txt", data2);
                write_to_file<tuple<double, double, double>>("3d_data3.txt", data3);
            }
            break;
            case 2:{
                cout<<"Enter column quantity:"<<endl;
                int n;
                cin>>n;
                cout<<"Enter column numbers:"<<endl;
                vector<int> columns;
                for(int i=0;i<n;i++){
                    int column;
                    cin>>column;
                    columns.push_back(column);
                }
                cout<<"Log or linear?"<<endl;
                bool log;
                cin>>log;
                plot_gnu(filename1,columns,columns_title,log);
                cout<<endl;
                }
            break;
            case 3:{
                
                Population<1, 10, 5> pop = {};
                int pop_size;
                double cros_p, mut_p;
                cout<<"Enter population size:"<<endl;
                cin>>pop_size;
                cout<<"Enter crossingover probability:"<<endl;
                cin>>cros_p;
                cout<<"Enter mutation probability:"<<endl;
                cin>>mut_p;
                pop.fill(pop_size);
                pop.setFunction([](double x) { return log(x)*cos(3*x-15); });
                pop.setPCross(cros_p);
                pop.setPMut(mut_p);
                cout<<"Starting simulation..."<<endl;
                cout<<"Enter 0 to stop or 1 to continue: "<<endl;
                int cont = 1;
                while (cont != 0)
                {
                    pop.step();
                    pop.draw();
                    cin>>cont;
                }
                cout<<"Simulation ended."<<endl;

                
            }
            break;
            case 9:
                return -1;
            break;

            default:
                n=-1;
                continue;

        }
    }
    return 0;
}