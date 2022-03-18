package com.drawing;

import java.util.ArrayList;
import java.util.List;

public class B_6 {

    public static void main(String[] args) {
        List<Object> array = new ArrayList<Object>();

        Object o = new Circle();
        array.add(o);
        o = new Square();
        array.add(o);
        o = new Triangle();
        array.add(o);
        o = new Square();
        array.add(o);

        int i = 1;
        int temp = 10;

        while (i < 5) {
            temp = temp % 3;
            Graphics.draw(array.get(temp));
            temp = temp + 2;
            i++;
        }
    }




}

/*
 *
 * What are the last three shape objects drawn by Main()?
 *
 * (a) triangle, square, circle
 * (b) square, square, triangle
 * (c) square, circle, square
 * (d) circle, triangle, square
 * (e) circle, square, triangle
 *
 */

