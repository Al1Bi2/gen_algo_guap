#pragma once

#include <vector>
#include <bitset>

namespace gen{
    namespace reproduction{
        class roulette{

        };
        class tournament{

        };
    }
    namespace crossover{
        class SBX{
            SBX(){};
            void do_crossingover(, double){
                
            }
        };
    }
    namespace genome{
        template<size_t lenght>
        class BGA{
            std::vector<bool> gen;
            std::bitset<lenght> genotype;
        };
        template<size_t lenght>
        class RGA{
        public:
            std::array<double,lenght> genotype;
            RGA():genotype(){};

        };
    }

    template <typename T>
    class population{
    public:
        
        population(): popul(),fitness(){};
        
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
}