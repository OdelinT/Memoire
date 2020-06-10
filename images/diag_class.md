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
        class tf_environment extends py_environment{
            {abstract} observation()
            {abstract} action()
            {abstract} reward()
        }
    }
    package agents{
        class dqn_agent{}
        class reinforce_agent{}
    }
}
package Memoire{
    class App{
        main()
    }
    package Agent{
            class MonAgent {}
    }
    package Environment{
        class MonEnvironment extends tf_environment{
            observation()
            action()
            reward()
        }
        class Place {
            + int id
            - int Size
            + int getFrequencation()
            + Dictionary<Product, int> quantitiesSold(Dictionary<Product, int> prices)
        }
        class Product {
            + int id
            - float BasePrice
        }
    }
}

note "Mais en fait, je peux parfaitement\nmodéliser mes classes par une matrice à\ndouble entrée (produit, prix), avec le\nprix en entrée, la quantité en sortie oO" as Remarque

Remarque ..up.. Place
Remarque ..up.. Product

Place "*" - "*" Product: vend >
MonEnvironment - "*" Place: comprend >

MonAgent -down- "2" MonEnvironment: instancie >

MonAgent --up-- "1" dqn_agent: utilise >
MonAgent --up-- "1" reinforce_agent: utilise > 

App -left- MonAgent: instancie, paramètre et mesure >

@enduml
~~~