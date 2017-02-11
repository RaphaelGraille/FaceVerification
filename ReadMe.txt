Raphaël Graille


        Projet de Traitement Vidéo et applications - INF6803 :

                 Reconnaissance de visages avec des CNN


1) Instalation des libraries

Le projet utilise les libraries Theano et Lasagne.

Pour les installer voir:
http://lasagne.readthedocs.org/en/latest/user/installation.html

Pour une installation "de zero" voir :
https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne

Pour accelérer le calcul j'ai également utilisé CudNN

2) Execution des tests

Pour executer les tests, lancer 'python test3precisions.py' en ligne de commande.


3) Entrainer un model

    - Sur la base de donné VGG
        Lancer 'python face_net.py' en ligne de commande

    - Sur une autre base de donnée
        Adapter la fonction 'load_database' a la nouvelle base de données
        Lancer 'python face_net.py' en ligne de commande

