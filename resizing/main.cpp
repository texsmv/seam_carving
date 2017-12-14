#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"

#include <iostream>
#include <vector>
#include <math.h>
#include <tuple>
#include <algorithm>
#include <list>
#include <stdlib.h>

using namespace cv;
using namespace std;

typedef pair<int, int> Punto;
typedef list<Punto> Camino;
typedef vector<vector<int>> Matriz;
typedef Vec3b Pixel;


/*
  Esta funcion calcula la distancia entre dos pixeles
  sumando la diferencia de cada canal
*/
int distancia_pixeles(Pixel& p1, Pixel& p2){
    return fabs(p1[0] - p2[0]) + fabs(p1[1] - p2[1]) + fabs(p1[2] - p2[2]);
}

/*
  Esta funcion retorna un pixel promedio de tres pixeles
  haciendoque cada canal sea el promedio de los otros tres
*/
Pixel promedio_tres(Pixel a, Pixel b, Pixel c){
    return Pixel((a[0] + b[0] + c[0])/3, (a[1] + b[1] + c[1])/3, (a[2] + b[2] + c[2])/3);
}

/*
  Esta funcion retorna un pixel promedio de dos pixeles
  haciendoque cada canal sea el promedio de los otros tres
*/
Pixel promedio_dos(Pixel a, Pixel b){
    return Pixel((a[0] + b[0])/2, (a[1] + b[1])/2, (a[2] + b[2])/2);
}

Mat reduce_image_r(Mat ima, Camino cam_min, int r)
{
    Mat new_image(r, ima.cols-1, CV_8UC3);
    list<pair<int,int>>::iterator it= cam_min.begin();
    for(int i=0; i<new_image.rows;i++)
    {
        int aux = 0;
        Vec3b* imgrow = ima.ptr<Vec3b>(i);
        Vec3b* mew_image = new_image.ptr<Vec3b>(i);
        for(int j = 0; j<new_image.cols;j++)
        {
            if(j == (*it).second)
                aux++;
            mew_image[j] = imgrow[j+aux];
        }
        it++;
    }
    return new_image;
}

Mat reduce_image_c(Mat ima, Camino cam_min)
{
    Mat new_image(ima.rows - 1,ima.cols,CV_8UC3);
    list<pair<int,int>>::iterator it=cam_min.begin();
    for(int j=0;j<new_image.cols;j++)
    {
        int aux = 0;
        for(int i=0;i<new_image.rows;i++)
        {
            if( i == (*it).first)
                aux++;
            new_image.at<Vec3b>(i,j) = ima.at<Vec3b>(i+aux,j);
        }
        it++;
    }
    return new_image;
}



class Imagen{
public:
    Imagen(){}
    Imagen(string path, string nombre){
        this->nombre = nombre;
        mat = imread(path, CV_LOAD_IMAGE_COLOR);
        energias = vector<vector<int>>(mat.rows, vector<int>(mat.cols, 0));
        energias_original = caminos = energias;
        mat_original= mat.clone();
        marcas = vector<vector<bool>>(mat.rows, vector<bool>(mat.cols, false));
    }
    void calcular_matriz_energias();
    void calcular_caminos_vertical();
    void calcular_caminos_horizontal();
    void colorear_camino_minimo_vertical(Camino*);
    void colorear_camino_minimo_horizontal(Camino*);
    void colorear_caminos_minimos_verticales(int n);
    void colorear_caminos_minimos_horizontales(int n);
    void clear_marcas();
    Camino* camino_minimo_vertical();
    Camino* camino_minimo_horizontal();
    vector<Camino*> caminos_minimos_verticales(int n);
    vector<Camino*> caminos_minimos_horizontales(int n);
    void duplicar_caminos_verticales(vector<Camino*>& caminos);
    void duplicar_caminos_horizontales(vector<Camino*>& caminos);
    void pixeles_derecha(int i, int j, vector<pair<int, int>>& pixeles, Matriz& energias);
    void pixeles_abajo(int i, int j, vector<pair<int, int>>& pixeles, Matriz& energias);
    void mostrar();
    void mostrar_final(int x, int y);
    void update();
    void redimensionar(int x, int y, int num_batch_x, int num_batch_y);

    ~Imagen(){}

    //matriz de pixeles de la imagen que se modificara
    Mat mat;
    //matriz de pixeles de la imagen original
    Mat mat_original;
    //matriz de energias(gradiente)
    Matriz energias;
    //copia de la matriz de energias(no se modificara)
    Matriz energias_original;
    //matriz que almacenara el camino minimo a cada pixel
    Matriz caminos;
    //nombre con el que guardara la imagen
    string nombre;
    //matriz de marcas que se usaran para evitar que los caminos se crucen
    vector<vector<bool>> marcas;
    //numero de pixeles vecinos que se tomaran en cuenta para las funciones pixeles_abajo y pixeles_derecha
    int k = 1;
};


void Imagen::redimensionar(int x, int y, int num_batch_x, int num_batch_y){
    // hallamos la varicion de pixeles que se desea en la imagen
    int dif_x = x - mat.cols;
    int dif_y = y - mat.rows;


    /*
      En este bucle realizamos de forma intercalada
      la eliminacion de caminos horizontales y verticales
    */

    // vert -> lo usamos para intercalar entre eliminar horizontal y vertical
    bool vert = true;

    while (dif_x < 0 || dif_y < 0) {
        calcular_matriz_energias();
        Camino* c_min;
        if(vert && dif_y < 0){
            // eliminamos un camino horizontal
            calcular_caminos_horizontal();
            c_min = camino_minimo_horizontal();
            mat = reduce_image_c(mat, *c_min);
            dif_y ++;
            // en caso de que haya que hacer cambios en el largo de la imagen intercalamos
            if(dif_x < 0)
                vert = false;
        }
        else{
            // eliminamos un camino vertical
            calcular_caminos_vertical();
            c_min = camino_minimo_vertical();
            mat = reduce_image_r(mat, *c_min, mat.rows);
            dif_x ++;
        `   vert = true;
        }
        /*
          con update volvemos a inicializar en ceros la matriz de energias y caminos
        */
        update();
        // limpiamos la matriz de marcas
        clear_marcas();
        // mostramos el cambios
        mostrar();
    }



    mostrar();

    vector<Camino*> caminos;
    int tam_batch_x = dif_x / num_batch_x;
    int tam_batch_y = dif_y / num_batch_y;


    clear_marcas();
    update();
    calcular_matriz_energias();

    /*
      en este bucle realizamos el redimensionamiento en altura
    */
    while(dif_y > 0){
        if(dif_y / tam_batch_y == 0)
            caminos = caminos_minimos_horizontales(dif_y);
        else
            caminos = caminos_minimos_horizontales(tam_batch_y);

        duplicar_caminos_horizontales(caminos);
        clear_marcas();
        update();
        mostrar();
        dif_y -= tam_batch_y;
    }

    mostrar();
    clear_marcas();
    update();
    calcular_matriz_energias();


    /*
      en este bucle realizamos el redimensionamiento en anchura
    */
    while(dif_x > 0){
        if(dif_x / tam_batch_x == 0)
          caminos = caminos_minimos_verticales(dif_x);
        else
            caminos = caminos_minimos_verticales(tam_batch_x);

        duplicar_caminos_verticales(caminos);
        clear_marcas();
        update();
        calcular_matriz_energias();
        mostrar();
        dif_x -= tam_batch_x;
    }

    // mostrmaos la imagen final, junto con la original y las guardamos
    mostrar_final(x, y);
}


void Imagen::clear_marcas(){
    marcas = vector<vector<bool>>(mat.rows, vector<bool>(mat.cols, false));
}


void Imagen::mostrar(){
    Mat copia = mat.clone();
    imshow("seam carving", copia);
    waitKey(1);
}

void Imagen::mostrar_final(int x, int y){
    Mat copia = mat.clone();
    Mat copia_redimensionada;
    Size size(x, y);
    resize(copia, copia_redimensionada,size);
    imshow("imagen original", mat_original);
    imshow("seam carving", copia);
    //imshow("redimensionado tradicional", copia_redimensionada);

    imwrite(nombre+"_seam_carving.jpg", copia);
    imwrite(nombre+"_redimensionado_tradicional.jpg", copia_redimensionada);
    waitKey(0);
}


void Imagen::update(){
    energias = vector<vector<int>>(mat.rows, vector<int>(mat.cols, 0));
    caminos = vector<vector<int>>(mat.rows, vector<int>(mat.cols, 0));
}

/*
Esta funcion calcula la gradiente de la imagenes
y la almacena en la matriz de energias y energias_original
*/
void Imagen::calcular_matriz_energias(){
    //iteramos en la matriz de la imagen
    for(int i = 0; i < mat.rows - 1; i++){
        // row -> fila actual
        Vec3b* row = mat.ptr<Vec3b>(i);
        // next_row -> siguiente fila
        Vec3b* next_row = mat.ptr<Vec3b>(i + 1);
        for(int j = 0; j < mat.cols - 1; j++){
            // comparacion de pixeles

            Vec3b pixel = row[j];
            Vec3b pixel_derecho = row[j + 1];
            Vec3b pixel_inferior = next_row[j];

            // aqui calculamos la energia de cada pizel y la almacenamos de la energia
            energias[i][j] = distancia_pixeles(pixel, pixel_derecho) + distancia_pixeles(pixel, pixel_inferior);
        }
    }

    for(int i = 0; i < mat.cols; i++){
        // completamos la ultima fila
        energias[mat.rows - 1][i] = energias[mat.rows - 2][i];
    }
    for(int i = 0; i < mat.rows; i++){
        // completamos la ultima columna
        energias[i][mat.cols - 1] = energias[i][mat.cols - 2];
    }
      // hacemos una copia de la matriz de energias antes de modificarlas
    energias_original = energias;
}

void Imagen::colorear_caminos_minimos_verticales(int n){
    vector<Camino*> caminos = caminos_minimos_verticales(n);

    auto it = caminos.begin();
    while(it != caminos.end()){
    colorear_camino_minimo_vertical((*it));
    it++;
    mostrar();
  }
}

void Imagen::colorear_caminos_minimos_horizontales(int n){
    vector<Camino*> caminos = caminos_minimos_horizontales(n);
    auto it = caminos.begin();
    while(it != caminos.end()){
    colorear_camino_minimo_horizontal((*it));
    it++;
    mostrar();
  }
}


/*
  retorna la coordena j y la energia acumulada de los k pixeles inferiores
  se penaliza la energia acumulada de los pixeles que esten marcados
*/
void Imagen::pixeles_abajo(int i, int j, vector<pair<int, int>>& pixeles, Matriz& energias){

    int inf, sup;
    inf = j - k;
    sup = j + k;
    if(inf < 0)
        inf = 0;
    if(sup >= mat.cols)
        sup = mat.cols - 1;
    for(int p = inf; p <= sup; p++){
        if(marcas[i + 1][p] == true)
            pixeles.push_back(make_pair(254*3 + energias[i + 1][p], p));
        else
            pixeles.push_back(make_pair(energias[i + 1][p], p));
    }
}

/*
  retorna la coordena i y la energia acumulada de los k pixeles del lado derecho
  se penaliza la energia acumulada de los pixeles que esten marcados
*/

void Imagen::pixeles_derecha(int i, int j, vector<pair<int, int>>& pixeles, Matriz& energias){

    int inf, sup;
    inf = i - k;
    sup = i + k;
    if(inf < 0)
        inf = 0;
    if(sup >= mat.rows)
        sup = mat.rows - 1;
    for(int p = inf; p <= sup; p++){
        if(marcas[p][j + 1] == true)
            pixeles.push_back(make_pair(254*3 + energias[p][j + 1], p));
        else
            pixeles.push_back(make_pair(energias[p][j + 1], p));
    }
}


void Imagen::duplicar_caminos_verticales(vector<Camino*>& caminos){
    //  esta variable contendra las posiciones, ordenadas y sin repetirse ,a duplicarse por cada fila
    vector< list<int>* > v_posiciones(mat.rows, nullptr);
    //  esta variable contendra el numero de veces que se repite cada posicion en cada fila
    vector< list<int>* > v_cantidades(mat.rows, nullptr);

    //
    for(int i = 0; i < mat.rows; i++){
        // posiciones tendra la i-aba posicion en j de cada camino
        vector<int> posiciones(caminos.size(), 0);
        for(int j = 0; j < caminos.size(); j++){
            posiciones[j] = caminos[j]->front().second;
            caminos[j]->pop_front();
        }

        // ordenammos las posiciones
        sort(posiciones.begin(), posiciones.end());

        //este lista contendra las posiciones ordenadas sin repetir
        list<int>* posiciones_nr = new list<int>();
        //este lista contendra el numero de veces que se repite cada posicion
        list<int>* cantidades = new list<int>();

        // comenzamos insertando la primera posicion
        posiciones_nr->push_back(posiciones[0]);
        cantidades->push_back(1);

        // iteramos en las posiiones y calculamos el numero de veces que se repiten
        for(int k = 1; k < posiciones.size(); k++){
            // si es una posicion ya ingresada aumentamos su v_cantidades
            if(posiciones[k] == posiciones_nr->back()){
                int temp = cantidades->back();
                cantidades->pop_back();
                cantidades->push_back(temp + 1);
            }
            // en caso de que sea un nuevo elemento lo insertamos en las posiciones y insertamos su cantidad
            else{
                posiciones_nr->push_back(posiciones[k]);
                cantidades->push_back(1);
            }
        }

        //almacenamos las posiciones sin repetir y ordenas, ademas de sus cantidades
        v_posiciones[i] = posiciones_nr;
        v_cantidades[i] = cantidades;

    }



    // creamos la matriz que contendra a la nueva imagen redimensionada

    Mat new_mat(mat.rows, mat.cols + caminos.size(), CV_8UC3);

    // iteramos sobre las filas
    for(int i = 0; i < new_mat.rows; i++){
        // cont -> nos servira para posicionar bien los pixeles
        int cont = 0;

        // pos -> iterador para las posiciones no repetidas de la fila i
        auto pos = v_posiciones[i]->begin();
        // cant -> iterador para las cantidades de las posiciones no repetidas de la fila i
        auto cant = v_cantidades[i]->begin();

        // iteramos sobre las columnas
        for(int j = 0; j < mat.cols; j++){
            // copiamos el pixel i, j en la nueva imagen teniendo en cuenta el aumento causado por los pixeles duplicados
            new_mat.at<Pixel>(i, j + cont) = mat.at<Pixel>(i, j);

            // si es que el pixel i, j debe duplicarse, lo copiamos el numero de veces que se repita
            if(j == *pos && pos != v_posiciones[i]->end()){
                for(int k = 0; k < *cant; k++){
                    // caso de la primera fila
                    if(i == 0)
                        new_mat.at<Pixel>(i, j + cont + k + 1) = promedio_dos(mat.at<Pixel>(i, j), mat.at<Pixel>(i + 1, j));
                    // caso de la ultima fila
                    else if(i == mat.rows - 1)
                        new_mat.at<Pixel>(i, j + cont + k + 1) = promedio_dos(mat.at<Pixel>(i, j), mat.at<Pixel>(i - 1, j));
                    // lo normal
                    else
                        new_mat.at<Pixel>(i, j + cont + k + 1) = promedio_tres(mat.at<Pixel>(i, j), mat.at<Pixel>(i - 1, j), mat.at<Pixel>(i, j));
                }
            // cont se incremente por el numero de pixeles que hayamos duplicado
            cont += *cant;
            // pasamos a la siguien posicion
            pos++;
            // pasamos a la siguien cantidad de posiciones
            cant++;
            }
        }
    }
    mat = new_mat;
}

void Imagen::duplicar_caminos_horizontales(vector<Camino*>& caminos){

    vector< list<int>* > v_posiciones(mat.cols, nullptr);
    vector< list<int>* > v_cantidades(mat.cols, nullptr);

    for(int i = 0; i < mat.cols; i++){
        vector<int> posiciones(caminos.size(), 0);
        for(int j = 0; j < caminos.size(); j++){
            posiciones[j] = caminos[j]->front().first;
            caminos[j]->pop_front();
        }
        // imprimir_v(&posiciones);
        sort(posiciones.begin(), posiciones.end());
        list<int>* posiciones_nr = new list<int>();
        list<int>* cantidades = new list<int>();
        posiciones_nr->push_back(posiciones[0]);
        cantidades->push_back(1);
        for(int k = 1; k < posiciones.size(); k++){
            if(posiciones[k] == posiciones_nr->back()){
                int temp = cantidades->back();
                cantidades->pop_back();
                cantidades->push_back(temp + 1);
            }
            else{
                posiciones_nr->push_back(posiciones[k]);
                cantidades->push_back(1);
            }
        }
        v_posiciones[i] = posiciones_nr;
        v_cantidades[i] = cantidades;

    }

    Mat new_mat(mat.rows + caminos.size(), mat.cols, CV_8UC3);

    for(int i = 0; i < new_mat.cols; i++){
        int cont = 0;

        auto pos = v_posiciones[i]->begin();
        auto cant = v_cantidades[i]->begin();
        for(int j = 0; j < mat.rows; j++){

            new_mat.at<Pixel>(j + cont, i) = mat.at<Pixel>(j, i);

            if(j == *pos && pos != v_posiciones[i]->end()){
                for(int k = 0; k < *cant; k++){

                    if(i == 0)
                        new_mat.at<Pixel>(j + cont + k + 1, i) = promedio_dos(mat.at<Pixel>(j, i), mat.at<Pixel>(j, i + 1));
                    else if(i == mat.cols - 1)
                        new_mat.at<Pixel>(j + cont + k + 1, i) = promedio_dos(mat.at<Pixel>(j, i), mat.at<Pixel>(j, i - 1));
                    else
                        new_mat.at<Pixel>(j + cont + k + 1, i) = promedio_tres(mat.at<Pixel>(j, i), mat.at<Pixel>(j, i - 1), mat.at<Pixel>(j, i));
                }
                cont += *cant;
                pos++;
                cant++;
            }

        }
    }
    mat = new_mat;
}


/*
  calculamos los caminos verticales para cada pixel
  todos los caminos empizan en la ultima fila
*/
void Imagen::calcular_caminos_vertical(){
    // iteramos desde la penultima fila hacia la primera fila
    for(int i = mat.rows - 2; i >= 0; i--){
        // iteramos desde la primera a la ultima columna
        for(int j = 0; j < mat.cols; j++){
            // pixeles -> hacemos que pixeles almacene los k pixeles bajo el pixel en la pasicion i, j
            vector< pair<int, int > > pixeles;
            pixeles_abajo(i, j, pixeles, energias);

            // almacenar el pixel del arreglo pixeles con menor energia acumulada
            auto minimo = *(min_element(pixeles.begin(), pixeles.end(), [](const pair<int, int> c1, const pair<int, int> c2){
                                                                                            return c1.first < c2.first;}));


            // guardar la posicion en el pixel i, j del pixel en caminos
            caminos[i][j] = minimo.second;
            // actualizar la energia del pixel en i , j
            energias[i][j] = minimo.first + energias[i][j];
      }
    }
}

void Imagen::calcular_caminos_horizontal(){
  // iteramos desde la penultima columna hacia la primera columna
    for(int j = mat.cols - 2; j >= 0; j--){
      // iteramos desde la primera a la ultima fila
        for(int i = 0; i < mat.rows; i++){
            // pixeles -> hacemos que pixeles almacene los k pixeles a la derecha del pixel en la pasicion i, j
            vector< pair<int, int > > pixeles;
            pixeles_derecha(i, j, pixeles, energias);


            // almacenar el pixel del arreglo pixeles con menor energia acumulada
            auto minimo = *(min_element(pixeles.begin(), pixeles.begin() + pixeles.size(), [](const pair<int, int> c1, const pair<int, int> c2){
                                                                                            return c1.first < c2.first;}));
            // guardar la posicion en el pixel i, j del pixel en caminos
            caminos[i][j] = minimo.second;
            // actualizar la energia del pixel en i , j
            energias[i][j] = minimo.first + energias[i][j];
      }
  }
}


Camino* Imagen::camino_minimo_horizontal(){
    // pare llamar a esta funcion es necesario primero haber llamado a la funcion calcular_caminos_vertical

    /*
      iteramos en la primera columna de la matriz de energias_original
      buscando la posicion con menor energia acumulada

    */

    // asumimos que el primer elemento es el menor

    // c_min -> posicion del menor camino
    int c_min = 0;
    // e_c_min -> energia acumulada del menor camino
    int e_c_min = energias[0][0];
    for(int i = 0; i < mat.rows; i++){
        // si e_c_min es mayor que la energia de la posicion actual la actualizamos
        if(e_c_min > energias[i][0]){
            e_c_min = energias[i][0];
            c_min = i;
        }
    }
    /*
      Una vez sabemos en que posicion de la primera columna esta el menor camino vertical
      reconstruimos el camino hasta la ultima columna de la imagen
    */
    // Camino -> esta es una lista de los puntos pertenecientes al camino


    Camino* cam_min = new Camino();
    for(int i = 0; i < mat.cols; i++){
        /*
         dejamos una marca en los caminos que ya usamos
        para evitar tener caminos pasando por el mismo pixel
        */
        marcas[c_min][i] = true;


        cam_min->push_back(make_pair(c_min, i));
        c_min = caminos[c_min][i];
    }
    return cam_min;
}

Camino* Imagen::camino_minimo_vertical(){
    // pare llamar a esta funcion es necesario primero haber llamado a la funcion calcular_caminos_horizontal

    /*
      iteramos en la primera fila de la matriz de energias_original
      buscando la posicion con menor energia acumulada

    */

    // asumimos que el primer elemento es el menor

    // c_min -> posicion del menor camino
    int c_min = 0;
    // e_c_min -> energia acumulada del menor camino
    int e_c_min = energias[0][0];

    for(int i = 0; i < mat.cols; i++){
        // si e_c_min es mayor que la energia de la posicion actual la actualizamos
        if(e_c_min > energias[0][i]){
            e_c_min = energias[0][i];
            c_min = i;
        }
    }

    /*
      Una vez sabemos en que posicion de la primera fila esta el menor camino vertical
      reconstruimos el camino hasta la ultima fila de la imagen
    */
    // Camino -> esta es una lista de los puntos pertenecientes al camino
    Camino* cam_min = new Camino();

    for(int i = 0; i < mat.rows; i++){
        /*
         dejamos una marca en los caminos que ya usamos
        para evitar tener caminos pasando por el mismo pixel
        */
        marcas[i][c_min] = true;


        cam_min->push_back(make_pair(i, c_min));
        c_min = caminos[i][c_min];
    }

    return cam_min;

}



vector<Camino*> Imagen::caminos_minimos_verticales(int n){
    // creamos un vector que almacenara los n caminos
    vector<Camino*> caminos_minimos;

    for(int i = 0; i < n; i++){
        // hacemos que la matriz de energias vuelva a su estado original
        energias = energias_original;
        /*
          calculamos la matriz de caminos verticales
          como no borramos las marcas del anterior caminos
          no se repetiran los mismo camino
        */
        calcular_caminos_vertical();
        //insertamos el camino al vector de caminos
        caminos_minimos.push_back( camino_minimo_vertical());
    }
    return caminos_minimos;
}



vector<Camino*> Imagen::caminos_minimos_horizontales(int n){
    // creamos un vector que almacenara los n caminos
    vector<Camino*> caminos_minimos;

    for(int i = 0; i < n; i++){
        // hacemos que la matriz de energias vuelva a su estado original
        energias = energias_original;
        /*
          calculamos la matriz de caminos horizontales
          como no borramos las marcas del anterior caminos
          no se repetiran los mismos camino
        */
        calcular_caminos_horizontal();
        //insertamos el camino al vector de caminos
        caminos_minimos.push_back( camino_minimo_horizontal());
    }
    return caminos_minimos;
}





void Imagen::colorear_camino_minimo_vertical(Camino* c_min){
    //cout<<c_min->front().first<<" "<<c_min->front().second<<endl;
    for(auto it : (*c_min)){
        mat.at<Pixel>(it.first, it.second) = Pixel(0, 0, mat.at<Pixel>(it.first, it.second)[2] + 100);
    }
}

void Imagen::colorear_camino_minimo_horizontal(Camino* c_min){
    //cout<<c_min->front().first<<" "<<c_min->front().second<<endl;
    for(auto it : (*c_min)){
        mat.at<Pixel>(it.first, it.second) = Pixel(0, 0, 254);
    }
}

int main( int argc, char** argv )
{

    cout<<"SEAM CARVING"<<endl;
    string nombre_imagen;
    int opcion_imagen = 0;
    cout<<"Ingrese el numero de la imagen"<<endl;
    cout<<"1.- playa"<<endl;
    cout<<"2.- lanchas"<<endl;
    cout<<"3.- torre de hercules"<<endl;
    cout<<"4.- paisaje"<<endl;
    cout<<"5.- ciudad"<<endl;
    cout<<"6.- lobo"<<endl;
    cout<<"7.- bicicleta"<<endl;
    cout<<"8.- capitan america"<<endl;
    cout<<"9.- avion"<<endl;
    cout<<"10.- turistas"<<endl;
    cout<<"11.- familia"<<endl;
    cout<<"12.- bote"<<endl;
    string path = "imagenes/";

    cin>>opcion_imagen;
    switch (opcion_imagen) {
        case 1: nombre_imagen = path + "playa.jpg";
        break;
        case 2: nombre_imagen = path + "lanchas.jpg";
        break;
        case 3: nombre_imagen = path + "torre_hercules.jpg";
        break;
        case 4: nombre_imagen = path + "paisaje.jpg";
        break;
        case 5: nombre_imagen = path + "ciudad.jpg";
        break;
        case 6: nombre_imagen = path + "lobo.jpg";
        break;
        case 7: nombre_imagen = path + "bicicleta.jpg";
        break;
        case 8: nombre_imagen = path + "capitan.jpg";
        break;
        case 9: nombre_imagen = path + "avion.jpg";
        break;
        case 10: nombre_imagen = path + "turistas.jpg";
        break;
        case 11: nombre_imagen = path + "familia.jpg";
        break;
        case 12: nombre_imagen = path + "bote.jpg";
        break;
        default: cout << "Usted ha ingresado una opcion incorrecta";
    }

    Imagen img(nombre_imagen, "output");

    cout<< "Dimension actual de la imagen : "<< img.mat.cols << " - "<<img.mat.rows<<endl;
    int x;
    int y;
    cout<<"Ingrese ingrese nueva dimension en x(largo)"<<endl;
    cin>>x;
    cout<<"Ingrese ingrese nueva dimension en y(ancho)"<<endl;
    cin>>y;

    int batch;
    cout<<"Numero de batchs"<<endl;
    cin>>batch;

    img.redimensionar(x, y, batch, batch);


    return 0;
}
