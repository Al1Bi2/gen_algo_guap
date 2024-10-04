#pragma once


namespace gen{
    namespace reproduction{
        class roulette{

        };
        class tournament{

        };
    }
    namespace genome{
        class BGA{

        };
        class RGA{

        };
    }

    template<typename T>
    class population{
    public:
        initialize(){

        }
        population(): population(),fitness(){};
        
    public:
        std::vector<genome::T> populat;
        std::vector<double> fitness;

    };
    template<typename T>
    class algo{
    public:
        algo(){};
        population<genome::T> p;
        


    };
}