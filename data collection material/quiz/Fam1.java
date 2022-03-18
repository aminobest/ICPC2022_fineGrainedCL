package com.drawing;

import java.util.ArrayList;
import java.util.List;

public class Fam1 {

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


       int limit = 10%4;

        for (int l=0; l<limit; l++) {
            Graphics.draw(array.get(l));
        }

    }

    /*
     *
     * What is the sequence of shape objects drawn by Main()?
     *
     */


}

