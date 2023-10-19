############### Projet recheche - Intelligence artificielle ###############
    ## reconnaissance automatique par classification naïve bayésienne ##

"""

Ce programme sert a réaliser une intelligence artificielle en
utilsant la méthode de classification bayésienne naive.

La base de données doit être un fichier excel (.xls)
Le fichier doit être "TYPE" et organisé de la façon suivante :
    1e Colonne : id (représentant le numéro de chaque fleur)
    2e Colonne : Nom_espece
    3e Colonne et plus : Les caractéristiques

    1e ligne : Nom des colonnes
    2e ligne et plus : l'ensemble des données

Projet réalisé par :
    Anthony BELLOTTO
    Océane GOUSSET
    Nathanaël LATOUR
    Aurore LE MERCIER
    Corentin ROCHE

"""

# Paramètres à modifier en fonction de chaque base de données.

Nom_excel = "IRIS.xls"  # Nom du fichier excel ou chemin pour l'ouvrir
Num_page = 0            # Numéro de la page de l'excel (0 pour la première, 1 pour la deuxième...)
alpha = 80              # Pourcentage de la part de la base de données appropriée pour l'apprentissage


# Modules importés

import xlrd         # exploitation du fichier excel
import math as m


############ CLASSES ############

class Espece_fleurs:
    # Classe qui représente les données de chaque espèce lors de l'apprentissage
    def __init__(self, data, id_, espece):
    # Initialisation des Variables

        self.data = data            # Contient l'ensemble des données des fleurs
        self.id_ = id_              # Contient l'ensemble des id des fleurs
        self.espece = espece        # Contient l'ensemble des espèces des fleurs
        self.nb_fleurs = len(data)  # Égale aux nombres de fleurs
        self.esper = []             # Va contenir les espérances de chaque espèce
        self.var = []               # Va contenir les variances de chaque espèce

    # Fonction liée à l'initialisation
        self.Proba__()

    def Proba__(self):
        # Fonction qui permet de calculer l'Espérance et la Variance de chaque espèce de fleurs
        global GLOBAL_Var
        for i in range(GLOBAL_Var[5]):
            self.esper.append(Esperance(self.data, i))
            self.var.append(Variance(self.data, i))


class Fleurs:
    # Classe qui représente une fleur de vérification
    def __init__(self, data, id_, espece):
    # Initialisation des Variables
        self.data = data            # Contient les données d'une fleur
        self.id_ = id_              # Contient l'id d'une fleur
        self.espece = espece        # Contient l'espèces d'une fleur
        self.proba = []             # Va contenir les probabilités que la fleur appartienne à une espèce
        self.espece_predi = 0       # Va contenir le nom de l'espèce que l'IA prédit
        self.valid = 0              # Variable de validation (1 si vrai et 0 si faux)

    # Fonctions liées à l'initialisation
        self.Proba__()
        self.Valid__()

    def Proba__(self):
        # Fonction qui calcule la proba que la fleur appartienne à chaque espèce
        global GLOBAL_Var, FLEURS_App
        evidence = 0
        for i in range(GLOBAL_Var[4]):
            p = 1/GLOBAL_Var[4]
            for j in range (GLOBAL_Var[4]):
                p *= Proba(self.data[j], FLEURS_App[i].esper[j], FLEURS_App[i].var[j])
            self.proba.append(p)
            evidence += p
        for i in range(GLOBAL_Var[4]):
            self.proba[i] /= evidence

    def Valid__(self):
        # Fonction qui permet de savoir si la fleur est bien de la bonne espèce
        global GLOBAL_Var
        index = Max_Proba(self.proba)
        self.espece_predi = GLOBAL_Var[6][index]
        if self.espece_predi == self.espece:
            self.valid = 1

############ FONCTION D'ANALYSE ############

def analyse_excel_App():
    # Exploitation des données d'apprentissage
    global GLOBAL_Var
    TOUTES_ESPECE = []

    for i in range(GLOBAL_Var[4]):
        data = []
        id_ = []
        for j in range(1, GLOBAL_Var[2]+1):
            if sheet.cell_value(j,1) == GLOBAL_Var[6][i]:
                d = []
                for k in range(GLOBAL_Var[5]):
                    d.append(float(sheet.cell_value(j, k+2)))
                data.append(d)
                id_.append(sheet.cell_value(j, 0))
        TOUTES_ESPECE.append(Espece_fleurs(data, id_, GLOBAL_Var[6][i]))
    return TOUTES_ESPECE

def analyse_excel_Verif():
    # Exploitation des données de vérification
    global GLOBAL_Var
    TOUTES_FLEURS = []

    for i in range(GLOBAL_Var[2]+1, GLOBAL_Var[1]+1):
        d = []
        for j in range(GLOBAL_Var[5]):
            d.append(float(sheet.cell_value(i, j+2)))
        TOUTES_FLEURS.append(Fleurs(d, sheet.cell_value(i, 0), sheet.cell_value(i, 1)))
    return TOUTES_FLEURS

def Nom_Espece(sheet, nb_fleurs):
    # Exploitation des noms d'espèces
    nom = []
    for i in range (1, nb_fleurs+1):
        if sheet.cell_value(i, 1) in nom :
            continue
        else :
            nom.append(sheet.cell_value(i, 1))
    return nom


############ FONCTIONS DE STATISTIQUE ############

def Esperance(data, col): # Pour E(X) = Somme(Xi)/n
    # Fonction qui calcule l'Espérance de chaque caractéristique d'une espèce
    som = 0
    for i in range(len(data)):
        som += data[i][col]
    return som/len(data)

def Esperance2(data, col): # Pour E(X**2) = Somme(Xi**2)/n
    # Fonction qui calcule l'Espérance2 de chaque caractéristique d'une espèce
    som = 0
    for i in range(len(data)):
        som += data[i][col]**2
    return som/len(data)

def Variance(data, col): # Pour V(X) = E(X**2)-E(X)**2
    # Fonction qui calcule la Variance de chaque caractéristique d'une espèce
    return Esperance2(data, col)-Esperance(data, col)**2

def Proba(val, esp, var):
    # Fonction qui calcule la probabilité
    inter_exp = (esp-val)**2
    return m.exp(-inter_exp/(2*var))/m.sqrt(2*m.pi*var)

def Max_Proba(prob):
    # Fonction qui renvoie l'index de la proba la plus élevée des différentes espèces
    p_max = prob[0]
    index = 0
    for i in range(1, len(prob)):
        if p_max < prob[i]:
            p_max = prob[i]
            index = i
    return index

def Arrondie(nb, d):
    # Fonction qui renvoie l'arrondie d'un nombre décimal
    nb = int(nb * 10**d)
    nb1 = int(nb * 10**(d-1)) * 10
    if nb-nb1 <= 5:
        nb += 1
    nb /= 10**d
    return nb


############ VARIABLES UTILES ############

book = xlrd.open_workbook(Nom_excel)    # Fonction qui ouvre le fichier excel
sheet=book.sheet_by_index(Num_page)     # Fonction qui ouvre la page de travail

nb_fleurs = sheet.nrows - 1                             # Nb de fleurs
nb_fleurs_apprentissage = int(nb_fleurs*0.8)            # Nb de fleurs pour l'apprentissage
nb_fleurs_verif = nb_fleurs - nb_fleurs_apprentissage   # Nb de fleurs pour la vérification
nb_caractéristique = sheet.ncols - 2                    # Nb de caractéristiques
nom_espece = Nom_Espece(sheet, nb_fleurs)               # Noms d'espéces différentes
nb_espece = len(nom_espece)                             # Nb d'espèces différentes

# Variable générale mis dans les fonctions qui en ont besoin
GLOBAL_Var = [sheet, nb_fleurs, nb_fleurs_apprentissage, nb_fleurs_verif, nb_espece, nb_caractéristique, nom_espece]
# global     [  0  ,  1  ,  2  ,  3  ,  4  ,  5  ,  6  ]

if __name__ == '__main__':      # Ce qui est exécuté uniquement quand le script est run depuis ce fichier

    # Apprentissage
    FLEURS_App = analyse_excel_App()

    # Vérification et visualisation des résultats
    FLEURS_Verif = analyse_excel_Verif()
    valid = 0
    for i in range(len(FLEURS_Verif)):
        print("Pour la fleur n°", i + nb_fleurs_apprentissage + 1," : \n\tLongeur des sépales : ", FLEURS_Verif[i].data[0],"\n\tLargeur des sépales : ", FLEURS_Verif[i].data[1],"\n\tLongueur des pétales : ",FLEURS_Verif[i].data[2], "\n\tLargueur des pétales : ",FLEURS_Verif[i].data[3],"\n\tEspèce : ",sheet.cell_value(i+nb_fleurs_apprentissage,1))
        print("\tLa classificateur prédit une fleurs de l'espèce : ",FLEURS_Verif[i].espece_predi)
        if FLEURS_Verif[i].valid == 1:
            valid += 1
            print("\tCette prédiction est juste !\n")
        else:
            print("\tCette prédiction est fausse !\n")

    pourcent = 100*valid/nb_fleurs_verif
    print("\nNotre classificateur Bayzienne naif a un ratio de ", Arrondie(pourcent, 1), "%\n\n")
    input("Appuyez sur Espace pour fermer la fenêtre !")
