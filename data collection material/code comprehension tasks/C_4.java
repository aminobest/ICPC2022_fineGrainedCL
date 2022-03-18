package com.drawing;

import java.util.ArrayList;
import java.util.List;

public class C_4 {

    public static void main(String[] args) {
        List<Object> array = new ArrayList<Object>();

        Object r = new Rectangle();
        array.add(r);
        Object e = new Triangle();
        Object s = new Circle();
        array.add(s);
        array.add(e);

        for(int i=0; i<array.size(); i = next(i,array)){
            Graphics.draw(array.get(i));
        }
    }

    public static int next(int i, List<Object> array) {
        if(array.get(i) instanceof Triangle) return array.size();
        else if(array.get(i) instanceof Rectangle) return i+2;
        return i-1;
    }


}
/*
 *
 * What is the sequence of shape objects drawn by Main()?
 *
 */