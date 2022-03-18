package com.drawing;

import java.util.ArrayList;
import java.util.List;

public class C_2 {

    public static void main(String[] args) {
        List<Object> array = new ArrayList<Object>();

        Object o = new Circle();
        array.add(o);
        o = new Rectangle();
        array.add(o);
        o = new Triangle();
        array.add(o);
        o = new Rectangle();
        array.add(o);
        o = new Circle();
        array.add(o);


        int c = 0;
        while (c < array.size()) {
            int nextIndex = array.size() % (c+1);
            Graphics.draw(array.get(nextIndex));
            c = c + 1;
        }
    }


}

/*
 *
 * What is the sequence of shape objects drawn by Main()?
 *
 */