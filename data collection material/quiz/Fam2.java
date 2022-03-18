package com.drawing;

import java.util.ArrayList;
import java.util.List;

public class Fam2 {

    public static void main(String[] args) {
        List<Object> array = new ArrayList<Object>();

        array.add(new Triangle());
        array.add(new Circle());
        array.add(new Square());
        array.add(new Circle());
        array.add(new Square());
        array.add(new Circle());
        array.add(new Triangle());
        array.add(new Square());


        for (int l=array.size()-1; l>0; l = l-2) {
            Graphics.draw(array.get(l));
        }

    }

    /*
     *
     * What are the last three shape objects drawn by Main()?
     *
     * (a) circle, circle, circle
     * (b) circle, circle,  square
     * (c) square, circle, square
     * (d) circle, triangle, square
     * (e) circle, square, triangle
     *
     */


}

