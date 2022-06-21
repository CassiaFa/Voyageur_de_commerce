import numpy as np
import pandas as pd
import tkinter as tk
import time
import random


class Lieu:
    """
    Lieu est une classe qui permet d'instancier un lieu ayant comme position avec une valeur x et y.
    
    __________________
    
    Attributs:
    ----------
        x: float
            la position en x du lieu
        y: float
            la position en y du lieu

    Méthodes:
    ---------
        distance(b): float
            retourne la distance euclidienne entre le lieu et le lieu b
    """

    def __init__(self, x, y):
        self.x = x # position en x du lieu
        self.y = y # position en y du lieu

    def distance(self, b):
        """
        Paramètres:
        -----------
            b: Lieu
                le lieu avec lequel on souhaite calculer la distance
        """

        distance = np.sqrt((self.x - b.x)**2 + (self.y - b.y)**2) # distance euclidienne entre le lieu et le lieu b

        return distance

class Graph:       
    """
    Graph est une classe qui permet d'instancier un objet graph dans lequel on retrouve une liste de lieux, et les coordonnées associés.

    __________________
    Attributs:
    ----------
        liste_lieux: list
            liste des lieux du graph
        coordonnees: list
            liste des coordonnées associées aux lieux
    
    Méthodes:
    ---------
        generation_lieux(nb_lieux, largeur, hauteur)
            génère les lieux du graph en fonction d'un nombre de lieux, d'une largeur et hauteur définit.
        
        calcul_matrice_cout_od()
            calcule la matrice de cout des OD du graph

        plus_proche_voisin(position, nb_lieux, dv)
            retourne le plus proche voisin d'une position donnée

        charger_graph(file)
            charge une liste de lieux et les coordonnées associées à partir d'un fichier .csv

        sauvegarder_graph()
            sauvegarde les coordonnées des lieux dans un fichier graph.csv
    """

    liste_lieux = [] # liste des lieux du graph
    coordonnees = [] # liste des coordonnées associées aux lieux

    @classmethod
    def generation_lieux(cls, nb_lieux, largeur, hauteur):
        """
        Paramètres:
        -----------
            nb_lieux: int
                nombre de lieux à générer
            largeur: int
                largeur de la zone de génération
            hauteur: int
                hauteur de la zone de génération
        """
        
        np.random.seed(1) # on fixe la seed pour avoir toujours le même aléatoire

        for i in range(nb_lieux):
            x = np.random.randint(0, largeur) # on génère un nombre aléatoire entre 0 et la largeur pour la coordonnée x
            y = np.random.randint(0, hauteur) # on génère un nombre aléatoire entre 0 et la hauteur pour la coordonnée y

            cls.liste_lieux.append(Lieu(x, y)) # on ajoute les lieux au graph
            cls.coordonnees.append([x, y]) # on ajoute les coordonnées au graph
        

    @classmethod
    def calcul_matrice_cout_od(cls):
        cls.matrice_od = np.zeros((len(cls.liste_lieux), len(cls.liste_lieux)))

        for i in range(len(cls.liste_lieux)):
            for j in range(i, len(cls.liste_lieux)):
                cls.matrice_od[i,j] = cls.liste_lieux[i].distance(cls.liste_lieux[j])
                cls.matrice_od[j,i] = cls.liste_lieux[i].distance(cls.liste_lieux[j])

        return cls.matrice_od
            

    @classmethod
    def plus_proche_voisin(cls, position, nb_lieux, dv):
        """
        Paramètres:
        -----------
            position: int
                position du lieu dont on souhaite trouver le plus proche voisin
            nb_lieux: int
                nombre de lieux du graph
            dv: int
                distance de voisinage
        """

        # Pour déterminer le chemin greedy
        ppv = np.argmin(np.ma.masked_array(cls.matrice_od[position], np.isin(list(range(nb_lieux)), dv)))

        return ppv

    @classmethod
    def charger_graph(cls, file):
        """
        Paramètres:
        -----------
            file: str
                nom du fichier .csv contenant les coordonnées des lieux
        """

        df = pd.read_csv(file) # on charge le fichier .csv
        cls.coordonnees = df.values # on récupère les coordonnées des lieux
        for i in cls.coordonnees: 
            cls.liste_lieux.append(Lieu(i[0], i[1])) # on ajoute les lieux au graph

    @classmethod
    def sauvegarder_graph(cls):

        cls.coordonnees = np.array(cls.coordonnees) # on convertit la liste de coordonnées en array
        np.savetxt("graph.csv", cls.coordonnees, delimiter=",", fmt="%i", header="x,y", comments='') # on sauvegarde les coordonnées dans un fichier .csv

class Route:
    """
    Route est une classe qui permet d'instencier un objet Route avec l'ordre des lieux à parcourir, ainsi que la distance total de la route.
    ______
    Attributs:
    ----------
        ordre: list
            liste des lieux à parcourir dans l'ordre
        distance: float
            distance totale de la route
    
    Méthodes:
    ---------
        calcul_distance(nb_lieux, matrice_p_pondere)
            calcule la distance totale de la route

        lieu_voisin_pondere(nb_lieux, matrice_p)
         
    """


    def __init__(self, depart = 0):
        self.ordre = [depart]
        self.distance = 0

    def calcul_distance_route(self, nb_lieux, matrice_p_pondere):
        """
        Paramètres:
        -----------
            nb_lieux: int
                nombre de lieux total utilisé
            matrice_p_pondere: numpy.ndarray
                matrice de cout pondérée des OD
        """
        
        position = self.ordre[0] # on récupère la position du premier lieu de la route

        for i in range(nb_lieux-1):

            # position = Graph.plus_proche_voisin(position, self.ordre)
            position = self.lieu_voisin_pondere(position, matrice_p_pondere) # on récupère un lieu voisin selon une pondération
            self.distance += Graph.matrice_od[self.ordre[i],position] # on ajoute la distance entre le lieu précédent et le lieu courant à la distance totale de la route
            
            self.ordre.append(position) # on ajoute le lieu courant à la route

        self.distance += Graph.matrice_od[self.ordre[-1],self.ordre[0]] # on ajoute la distance entre le dernier lieu de la route et le lieu de départ
        self.ordre.append(self.ordre[0]) # on ajoute le premier lieu à la route pour fermer la route

    def lieu_voisin_pondere(self, position, matrice_p):
        """
            Méthode permetant de déterminer le lieu voisin choisie par la fourmi

            Paramètres:
            -----------
                position: int
                    position du lieu courant
                matrice_p: numpy.ndarray
                    matrice de cout pondérée des OD
        """

        vecteur_visibilite = (1/Graph.matrice_od[position])**b # on calcule le vecteur de visibilité, connaitre la pondération de chaque lieu (probabilité d'être selectionné)
        vecteur_pp = matrice_p[position]**a # on calcule le vecteur de pondération
        destination = vecteur_pp*vecteur_visibilite # on calcule le vecteur destination
        # destination[np.isinf(destination)] = 0 # on supprime les valeurs infini
        destination[np.ma.masked_array(destination, np.isin(list(range(nb_lieux)), self.ordre)).mask] = 0 # on supprime les valeurs déjà utilisées

        destination = destination/destination.sum() # on normalise le vecteur destination, pour avoir une probabilité uniforme
        test = destination.tolist() # on convertit le vecteur destination en liste

        return np.random.choice(np.ma.masked_array(list(range(nb_lieux)), np.isin(list(range(nb_lieux)), self.ordre)), 1, p=test)[0]

    def __eq__(self, other):
        return self.ordre == other.ordre

    def __gt__(self, other):
        return self.distance > other.distance

    def __str__(self):
        return f"Ordre = {self.ordre}, \nDistance : {self.distance:.2f}"

class Affichage:
    pass

class TSP_ACO:
    """
    TSP_ACO est une classe qui permet de lancer l'algorithme ACO pour le problème du TSP.
    ______

    Attributs:
    ----------
        best_route: Route
            route optimale trouvée
        
    Méthodes:
    ---------
        calculer_circuit_fourmis()
            calcule la route optimale pour le problème du TSP avec l'algorithme ACO
    """

    # formule phéromone
    #   - dépot des phéromone
    #   - volatilisation des phéromones
    # Distance matrice_od
    # Comparaison route (__eq__, __lt__, __gt__, __le__, __ge__, __repr__, __neq__)
    best_route = None
    quantite_pheromone = 200 # quantité de phéromone déposée
    taux_evaporation = 0.5 # taux d'évaporation des phéromones

    condition = 0
    nb_iterations = 0
    
    @classmethod
    def calculer_meilleur_route(cls):
        """
        Méthode permetant de calculer la route optimale pour le problème du TSP avec l'algorithme ACO
        """       

        # on initialise la matrice de phéromone
        matrice_p = np.ones((nb_lieux,nb_lieux))/100


        while cls.condition < 0.5:
            # générer les lieux
            # nombre de fourmis dans lieu (i)
            # fourmi choisi ville de destination j de la liste ville à visité
            # liste dépendante à chaque fourmis

            fourmis_route = []
            
            cls.condition = 0
            cls.calculer_circuit_fourmis(fourmis_route, matrice_p)

            # mise à jour des phéromones
            matrice_p = cls.mise_a_jour_pheromone(matrice_p, fourmis_route)

            cls.nb_iterations += 1

            print(cls.nb_iterations)
            print(cls.condition)

    @classmethod
    def calculer_circuit_fourmis(cls, fourmis_route, matrice_p):

        for k in range(m):
            depart = np.random.randint(nb_lieux) # à adapater pour éviter qu'un lieu ne soit jamais sélectionné
            r = Route(depart)
            r.calcul_distance_route(nb_lieux, matrice_p)

            fourmis_route.append(r)

            if cls.best_route is None :
                cls.best_route = r

            elif cls.best_route == r:
                if cls.best_route > r:
                    cls.best_route = r
                
                cls.condition += 1

            elif cls.best_route > r:
                cls.best_route = r
                cls.condition = 0 # si best route est changé, on réinitialise la condition

            # print(f" Fourmi {k} \n------------\nordre : {r.ordre} \ndistance : {r.distance} \n============")
        
        cls.condition /= m


    @classmethod
    def mise_a_jour_pheromone(cls, matrice_p, fourmis_route):
        """
        Méthode permetant de mettre à jour la matrice de phéromone
        """

        matrice_ajout = np.zeros((nb_lieux,nb_lieux))

        for f in fourmis_route:
            for t in range(len(f.ordre)-1):
                matrice_ajout[f.ordre[t],f.ordre[t+1]] += cls.quantite_pheromone / f.distance
                

        matrice_p = cls.taux_evaporation * matrice_p + matrice_ajout
        
        return matrice_p

         

LARGEUR = 800
HAUTEUR = 600

nb_lieux = 10
m = nb_lieux # nombre de fourmis
a = 2 # paramètre d'importance des phéromones
b = 10 # paramètre d'importance des visibilité des lieux

Graph.generation_lieux(nb_lieux, LARGEUR, HAUTEUR)
# Graph.charger_graph("./graph_5.csv")
# print(Graph.coordonnees)
print(Graph.calcul_matrice_cout_od())
print("==========================")
# print(Graph.plus_proche_voisin())
# Graph.sauvegarder_graph()
# r1 = Route(nb_lieux)
# print(f"ordre : {r1.ordre} \ndistance : {r1.distance}")

TSP_ACO.calculer_meilleur_route()

print(TSP_ACO.best_route)