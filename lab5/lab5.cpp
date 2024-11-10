#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <utility>
using State = int;
using Output = int;
class FSM{
public:
    FSM():current_state(0),LDtable(){};
    FSM(std::initializer_list<char> input_chars,
        int states_quantity,
        std::initializer_list<std::pair<State,Output>> cells):current_state(0){
            for(auto& ch: input_chars){
                LDtable[ch] = {};
                for(int state = 0; state < states_quantity; state ++){
                    
                }
            }
        }
    std::pair<State,Output> do_step(char input_char){
        auto next = LDtable.at(input_char).at(current_state);
        current_state = next.first;
        return next;
    }
private:
     int current_state = State(-1);
    std::map<char,std::vector<std::pair<State,Output>>> LDtable;
};

class RegEx{
public:
    RegEx(){};
    RegEx(std::initializer_list<std::string> regex_string_lists){
        generate_FSM(regex_string_lists);

    }

private:
    FSM fsm;
    void generate_FSM(std::initializer_list<std::string> regex_string){
        std::vector<std::pair<char,std::vector<int>>> divided_regex;
        int char_counter = 1;
        for(auto& string : regex_string){
            divided_regex.push_back({'|',{0}});
            for(auto& ch: string){
                if(divided_regex.at(divided_regex.size()-1).first!='|'){
                    divided_regex.push_back({'|',{}});
                }
                divided_regex.push_back({ch,{char_counter}});
                char_counter++;
            }
            divided_regex.push_back({'|',{}});
        }

    }
};


int main(){
    RegEx r{"<ab>","<abc>"};
    return 0;
}