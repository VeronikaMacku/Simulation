"""
FINITE STATE MACHINE
Accepts all strings of type abab(abab)*
"""
class ababMachine():
    def __init__(self):
        self.init_state = "START"
        self.final_state = "b2"
        self.current_state = self.init_state

    STATES = [  "START",
                "a1", 
                "b1",
                "a2",
                "b2",
                "ERROR"
    ]

    INPUTS = [  "a",
                "b",
                "not a",
                "not b"
    ]

    #STATE_TRANSITION_TABLE
    TRANSITIONS = {
        "START":{"a":"a1","not a":"ERROR"},
        "a1":{"b":"b1","not b":"ERROR"},
        "b1":{"a":"a2","not a":"ERROR"},
        "a2":{"b":"b2","not b":"ERROR"},
        "b2":{"a":"a1","not a":"ERROR"},
        "ERROR":{"a":"ERROR","not a":"ERROR","b":"ERROR","not b":"ERROR"} 
    }

    def from_state_to_state(self, input):
        print("-----------------")
        print(f"Current state is:{self.current_state}")
        print(f"Possible inputs:{list(self.TRANSITIONS[self.current_state])}")
        print(f"Current input: {input}")

        state_transitions = self.TRANSITIONS[self.current_state]
        if input in state_transitions:
            self.current_state = state_transitions[input]
        else:
            self.current_state = state_transitions[list(state_transitions)[1]]  #Not the expected char, input according to 'not char' at pos 1
        print(f"New state is:{self.current_state}")
        
    def evaluate(self, string):
        for i in range(0,len(string)):
            input = string[i]
            self.from_state_to_state(input)

            if self.current_state == "ERROR":
                break

        print("-----------------")
        if self.current_state == self.final_state:
            print("ACCEPTED.")
        else: 
            print("REJECTED.")

        print(" ")
        self.reset()

    def reset(self):
        self.current_state = self.init_state

fsMachine = ababMachine()

while True:
    print("This finite state machine accepts only strings of type abab(abab)*.")
    string_to_Eval = input("Write a string to enter into machine or 0 to quit the program: ")
    if string_to_Eval == "0":
        break
    fsMachine.evaluate(string_to_Eval)
