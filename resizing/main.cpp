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


int distancia_pixeles(Pixel& p1, Pixel& p2){
    return fabs(p1[0] - p2[0]) + fabs(p1[1] - p2[1]) + fabs(p1[2] - p2[2]);
}

Pixel promedio1(Pixel a, Pixel b, Pixel c){
    return Pixel((a[0] + b[0] + c[0])/3, (a[1] + b[1] + c[1])/3, (a[2] + b[2] + c[2])/3);
}

Pixel promedio2(Pixel a, Pixel b){
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
    void eliminar_camino_minimo_vertical();
    void colorear_camino_minimo_vertical(Camino*);
    void eliminar_camino_minimo_horizontal();
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
    void save();
    void redimensionar(int x, int y, int num_batch_x, int num_batch_y);
    Mat mostrar_copia();
    ~Imagen(){}

    Mat mat;
    Mat mat_original;
    Matriz energias;
    Matriz energias_original;
    Matriz caminos;
    string nombre;
    vector<vector<bool>> marcas;
    int k = 1;
};


void Imagen::redimensionar(int x, int y, int num_batch_x, int num_batch_y){
    int dif_x = x - mat.cols;
    int dif_y = y - mat.rows;

    bool vert = true;
    int cont = 0;
    while (dif_x < 0 || dif_y < 0) {
        calcular_matriz_energias();
        Camino* c_min;
        if(vert && dif_y < 0){
            calcular_caminos_horizontal();
            c_min = camino_minimo_horizontal();
            mat = reduce_image_c(mat, *c_min);
            dif_y ++;
            if(dif_x < 0)
                vert = false;
        }
        else{
            calcular_caminos_vertical();
            c_min = camino_minimo_vertical();
            mat = reduce_image_r(mat, *c_min, mat.rows);
            dif_x ++;
        vert = true;
        }
        update();
        clear_marcas();
        if(cont == 3){
            mostrar();
            cont = 0;
        }
        cont++;
    }



    mostrar();

    vector<Camino*> caminos;
    int tam_batch_x = dif_x / num_batch_x;
    int tam_batch_y = dif_y / num_batch_y;
    //cout<<dif_x<<"  "<<dif_y<<endl;
    //cout<<tam_batch_x<<"  "<<tam_batch_y<<endl;
    clear_marcas();
    update();
    calcular_matriz_energias();
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

    while(dif_x > 0){
        if(dif_x / tam_batch_x == 0)
        caminos = caminos_minimos_verticales(dif_x);
        else
            caminos = caminos_minimos_verticales(tam_batch_x);
        duplicar_caminos_verticales(caminos);
        clear_marcas();
        update();
        mostrar();
        dif_x -= tam_batch_x;
    }
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

void Imagen::save(){
    imwrite(nombre+"_final.jpg", mat);
}

void Imagen::update(){
    energias = vector<vector<int>>(mat.rows, vector<int>(mat.cols, 0));
    caminos = energias;
}

void Imagen::calcular_matriz_energias(){
    for(int i = 0; i < mat.rows - 1; i++){
        Vec3b* row = mat.ptr<Vec3b>(i);
        Vec3b* next_row = mat.ptr<Vec3b>(i + 1);
        for(int j = 0; j < mat.cols - 1; j++){
            Vec3b pixel = row[j];
            Vec3b pixel_derecho = row[j + 1];
            Vec3b pixel_inferior = next_row[j];
            energias[i][j] = distancia_pixeles(pixel, pixel_derecho) + distancia_pixeles(pixel, pixel_inferior);
        }
    }
    for(int i = 0; i < mat.cols; i++){
        energias[mat.rows - 1][i] = energias[mat.rows - 2][i];
    }
    for(int i = 0; i < mat.rows; i++){
        energias[i][mat.cols - 1] = energias[i][mat.cols - 2];
    }
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

    auto imprimir = [](list<int>* lista){
        auto it = lista->begin();
        while(it != lista->end()){
            cout<<*it<<" - ";
            it++;
        }
        cout<<endl;
    };

    auto imprimir_v = [](vector<int>* lista){
        auto it = lista->begin();
        while(it != lista->end()){
            cout<<*it<<" - ";
            it++;
        }
        cout<<endl;
    };

    vector< list<int>* > v_posiciones(mat.rows, nullptr);
    vector< list<int>* > v_cantidades(mat.rows, nullptr);

    for(int i = 0; i < mat.rows; i++){
        vector<int> posiciones(caminos.size(), 0);
        for(int j = 0; j < caminos.size(); j++){
            posiciones[j] = caminos[j]->front().second;
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

        // imprimir(v_posiciones[i]);
        // imprimir(v_cantidades[i]);
        // cout<<endl;
        // waitKey(0);
    }

    Mat new_mat(mat.rows, mat.cols + caminos.size(), CV_8UC3);

    for(int i = 0; i < new_mat.rows; i++){
        int cont = 0;
        Vec3b* row = mat.ptr<Vec3b>(i);
        Vec3b* row_last;
        Vec3b* row_next;
        if(i != 0)
            row_last = mat.ptr<Vec3b>(i - 1);
        if(i != mat.rows - 1)
            row_next = mat.ptr<Vec3b>(i + 1);
        Vec3b* new_row = new_mat.ptr<Vec3b>(i);
        auto pos = v_posiciones[i]->begin();
        auto cant = v_cantidades[i]->begin();
        for(int j = 0; j < mat.cols; j++){
            new_row[j + cont] = row[j];
            if(j == *pos){
                for(int k = 0; k < *cant; k++){
                    // new_row[j + cont + k + 1] = row[j];
                    if(i == 0)
                        new_row[j + cont + k + 1] = promedio2(row[j], row_next[j]);
                    else if(i == mat.rows - 1)
                        new_row[j + cont + k + 1] = promedio2(row[j], row_last[j]);
                    else
                        new_row[j + cont + k + 1] = promedio1(row[j], row_last[j], row_next[j]);
                }
            cont += *cant;
            pos++;
            cant++;
            }
        }
    }
    mat = new_mat;
}

void Imagen::duplicar_caminos_horizontales(vector<Camino*>& caminos){

    auto imprimir = [](list<int>* lista){
        auto it = lista->begin();
        while(it != lista->end()){
            cout<<*it<<" - ";
            it++;
        }
        cout<<endl;
    };

    auto imprimir_v = [](vector<int>* lista){
        auto it = lista->begin();
        while(it != lista->end()){
            cout<<*it<<" - ";
        it++;
        }
        cout<<endl;
    };

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

    // imprimir(v_posiciones[i]);
    // imprimir(v_cantidades[i]);
    // cout<<endl;
    // waitKey(0);
    }

    Mat new_mat(mat.rows + caminos.size(), mat.cols, CV_8UC3);

    for(int i = 0; i < new_mat.cols; i++){
        int cont = 0;

        auto pos = v_posiciones[i]->begin();
        auto cant = v_cantidades[i]->begin();
        for(int j = 0; j < mat.rows; j++){
            new_mat.at<Pixel>(j + cont, i) = mat.at<Pixel>(j, i);
            if(j == *pos){
                for(int k = 0; k < *cant; k++){
                    // new_row[j + cont + k + 1] = row[j];
                    if(i == 0)
                        new_mat.at<Pixel>(j + cont + k + 1, i) = promedio2(mat.at<Pixel>(j, i), mat.at<Pixel>(j, i + 1));
                    else if(i == mat.rows - 1)
                        new_mat.at<Pixel>(j + cont + k + 1, i) = promedio2(mat.at<Pixel>(j, i), mat.at<Pixel>(j, i - 1));
                    else
                        new_mat.at<Pixel>(j + cont + k + 1, i) = promedio1(mat.at<Pixel>(j, i), mat.at<Pixel>(j, i - 1), mat.at<Pixel>(j, i));
                }
                cont += *cant;
                pos++;
                cant++;
            }
        }
    }
    mat = new_mat;
}

void Imagen::calcular_caminos_vertical(){

    for(int i = mat.rows - 2; i >= 0; i--){
        for(int j = 0; j < mat.cols; j++){
            vector< pair<int, int > > pixeles;
            pixeles_abajo(i, j, pixeles, energias);
            auto minimo = *(min_element(pixeles.begin(), pixeles.begin() + pixeles.size(), [](const pair<int, int> c1, const pair<int, int> c2){
                                                                                            return c1.first < c2.first;}));
            caminos[i][j] = minimo.second;
            energias[i][j] = minimo.first + energias[i][j];
      }
    }
}

void Imagen::calcular_caminos_horizontal(){

    for(int j = mat.cols - 2; j >= 0; j--){
        for(int i = 0; i < mat.rows; i++){

            vector< pair<int, int > > pixeles;
            pixeles_derecha(i, j, pixeles, energias);
            auto minimo = *(min_element(pixeles.begin(), pixeles.begin() + pixeles.size(), [](const pair<int, int> c1, const pair<int, int> c2){
                                                                                            return c1.first < c2.first;}));
            caminos[i][j] = minimo.second;
            energias[i][j] = minimo.first + energias[i][j];
      }
  }
}


Camino* Imagen::camino_minimo_horizontal(){

    int c_min = 0;
    int e_c_min = energias[0][0];
    for(int i = 0; i < mat.rows; i++){
        if(e_c_min > energias[i][0]){
            e_c_min = energias[i][0];
            c_min = i;
        }
    }
    Camino* cam_min = new Camino();
    for(int i = 0; i < mat.cols; i++){
        marcas[c_min][i] = true;
        cam_min->push_back(make_pair(c_min, i));
        c_min = caminos[c_min][i];
    }
    return cam_min;
}

Camino* Imagen::camino_minimo_vertical(){
    int c_min = 0;
    int e_c_min = energias[0][0];
    for(int i = 0; i < mat.cols; i++){
        if(e_c_min > energias[0][i]){
            e_c_min = energias[0][i];
            c_min = i;
        }
    }
    Camino* cam_min = new Camino();
    for(int i = 0; i < mat.rows; i++){
        marcas[i][c_min] = true;
        cam_min->push_back(make_pair(i, c_min));
        c_min = caminos[i][c_min];
    }

    return cam_min;

}

vector<Camino*> Imagen::caminos_minimos_verticales(int n){
    vector<Camino*> caminos_minimos;
    calcular_matriz_energias();
    energias_original = energias;
    for(int i = 0; i < n; i++){
        energias = energias_original;
        calcular_caminos_vertical();
        caminos_minimos.push_back( camino_minimo_vertical());
        update();
    }
    return caminos_minimos;
}



vector<Camino*> Imagen::caminos_minimos_horizontales(int n){
    vector<Camino*> caminos_minimos;
    calcular_matriz_energias();
    energias_original = energias;
    for(int i = 0; i < n; i++){
        energias = energias_original;
        calcular_caminos_horizontal();
        caminos_minimos.push_back( camino_minimo_horizontal());
        update();
    }
    return caminos_minimos;
}


void Imagen::eliminar_camino_minimo_vertical(){
    Camino* c_min = camino_minimo_vertical();
    mat = reduce_image_r(mat, *c_min, mat.rows);
}

void Imagen::eliminar_camino_minimo_horizontal(){
    Camino* c_min = camino_minimo_horizontal();
    mat = reduce_image_c(mat, *c_min);
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

    img.save();
    img.mostrar();
    waitKey(0);
    return 0;
}
