~~~plantuml
@startuml
skinparam monochrome true
skinparam shadowing false

package tf_agents{
    package environments{
        class py_environment{
            {abstract} observation()
            {abstract} action()
            {abstract} reward()
        }
    }
    package agents{
        class dqn_agent{}
        class reinforce_agent{}
        class sac_agent{}
    }
}
package Memoire{
    class Test{
        main()
    }
    package Environment{
        class MonEnvironment extends py_environment{
            observation()
            action()
            reward()
        }
    }
}

Test -left- MonEnvironment: instancie >
Test --up-- dqn_agent: applique >
Test --up-- sac_agent: applique >
Test --up-- reinforce_agent: applique >

@enduml
~~~