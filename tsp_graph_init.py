import numpy as np
import pandas as pd
import tkinter as tk
import time
import random


class Lieu:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, b):
        distance = np.sqrt((self.x - b.x)**2 + (self.y - b.y)**2)

        return distance

class Graph:       

    coordonnees = []
    liste_lieux = []

    @classmethod
    def generation_lieux(cls, nb_lieux, largeur, hauteur):
        np.random.seed(1) 

        for i in range(nb_lieux):
            x = np.random.randint(0, largeur)
            y = np.random.randint(0, hauteur)

            cls.coordonnees.append([x, y])
            cls.liste_lieux.append(Lieu(x, y))
        

    @classmethod
    def calcul_matrice_cout_od(cls):
        cls.matrice_od = np.zeros((len(cls.liste_lieux), len(cls.liste_lieux)))

        for i in range(len(cls.liste_lieux)):
            for j in range(i, len(cls.liste_lieux)):
                cls.matrice_od[i,j] = cls.liste_lieux[i].distance(cls.liste_lieux[j])
                cls.matrice_od[j,i] = cls.liste_lieux[i].distance(cls.liste_lieux[j])

        return cls.matrice_od
            

    @classmethod
    def plus_proche_voisin(cls, position, dv):
        # ppv = np.argmin(np.ma.masked_where(cls.matrice_od==0 , cls.matrice_od[position]) , axis=1)
        ppv = np.argmin(np.ma.masked_array(cls.matrice_od[position], np.isin(list(range(nb_lieux)), dv)))
        return ppv

    @classmethod
    def charger_graph(cls, file):
        df = pd.read_csv(file)
        cls.coordonnees = df.values
        for i in cls.coordonnees:
            cls.liste_lieux.append(Lieu(i[0], i[1]))

    @classmethod
    def sauvegarder_graph(cls):
        
        cls.coordonnees = np.array(cls.coordonnees)
        np.savetxt("graph.csv", cls.coordonnees, delimiter=",", fmt="%i", header="x,y", comments='')


class Route:
    
    def __init__(self, nb_lieux):
        self.ordre = [0]
        self.distance = 0
        
        self.calcul_distance_route(nb_lieux)


    def calcul_distance_route(self, nb_lieux):
        position = self.ordre[0]

        for i in range(nb_lieux-1):

            position = Graph.plus_proche_voisin(position, self.ordre)
            self.distance += Graph.matrice_od[self.ordre[i],position]
            print(Graph.matrice_od[self.ordre[i],position])
            
            self.ordre.append(position)
        self.ordre.append(0)


class Affichage:
    pass

class TSP_ACO:
    pass

LARGEUR = 800
HAUTEUR = 600

nb_lieux = 5

Graph.generation_lieux(nb_lieux, LARGEUR, HAUTEUR)
# Graph.charger_graph("./graph_5.csv")
# print(Graph.coordonnees)
print(Graph.calcul_matrice_cout_od())
# print(Graph.plus_proche_voisin())
# Graph.sauvegarder_graph()
r1 = Route(nb_lieux)
print(f"ordre : {r1.ordre} \ndistance : {r1.distance}")