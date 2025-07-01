// Code for the "Eval" class, which implements the "gef" function in the paper
// "Multi-robot Motion Planning based on Nets-within-Nets Modeling and Simulation"
// Sofia Hustiu, Eva Robillard, Joaquín Ezpeleta, Cristian Mahulea and Marius Kloetzer
// To compile all the required files, just execute "javac *.java"

import java.util.Set;
import java.util.Iterator;
import java.util.TreeMap;
import java.util.HashSet;
import java.util.HashMap;
import java.util.SortedMap;
import java.util.LinkedHashMap;
import java.util.ArrayList;
import java.util.Scanner;

public class Eval {
    // P-indexed: partition occupacy multiset
    private static String partitionOccStr;                  
    private static SMultiSet partitionOccBag;

    // P-indexed: partition capacity multiset; "4'p1,1'p2,1'p3,1'p4,1'p5"
    private static String capStr;                  
    private static SMultiSet capBag;

    // Y-indexed: "h" mapping. For a given p\in P, h(p) is the
    // set of regions to which p belongs
    // "p1,b4|p5,b1|...|p4,b1,b2"
    private static String hStr;                  
    private static SortedMap<String,Set<String>> hBag;

    // set B of boolean variables. They will be extracted from "hBag"
    private static Set<String> B;

    // set P of partitions. They will be extracted from "capBag"
    private static Set<String> P;

    static final String SEP = "'";

    static long numTrues; //number of invocations to "gef" with a positive results. For monitoring purposes
    static long numGefs; //number of invocations to "gef". For monitoring purposes
    //---------------------------------------------------------------------------
    // There some performance problems when the system is big. Any firing requieres checking
    // a lot of transitions, and form each conver string multisets into multisets. And the set of string
    // multisets is quite reduced, but its conversion is made many times. It seems interesting to avoid
    // such conversion repetitions. For that, we will we are going to use a dictionary
    // <string_multi_set,multi_set>.
    private static LinkedHashMap<String,SMultiSet> multiSetDict;
    //---------------------------------------------------------------------------
    // Initialization function with initial global capacity, region occupancy
    // and h function
    public Eval(String cap,String part,String h) {
        capStr = cap;
        partitionOccStr = part;

        multiSetDict = new LinkedHashMap<String,SMultiSet>();

        capBag = new SMultiSet(cap);
        partitionOccBag = new SMultiSet(partitionOccStr);

        hStr = h;
        hBag = h2bag(hStr);

        // compute B set
        B = new HashSet<String>();
        for (SortedMap.Entry<String,Set<String>> entry : hBag.entrySet()) {
            B.addAll(entry.getValue());
        }

        // compute P set
        P = capBag.keySet();
        numTrues = 0;
        numGefs = 0;
    }
    //---------------------------------------------------------------------------
    // If key in multiSetDict, it returns the multiset form. If not, it creates and
    // insert the multiset form of key into "multiSetDict", and returns it
    private static SMultiSet getMultiFromString(String key) {
        SMultiSet ms = null;

        if (multiSetDict.containsKey(key)) {
            ms = multiSetDict.get(key);
        }
        else {
            ms = new SMultiSet(key);
            multiSetDict.put(key,ms);
        }

        assert ms != null:"Could not find or insert multiset";
        return ms;
    } 
    //---------------------------------------------------------------------------
    //"hS" is of the form "p1,b4|p5,b1|p2,b3|p3,b2|p4,b1,b2"
    // returns a map such that res[p4]={b1,b2},res[p1]={b4},...
    public static SortedMap<String,Set<String>> h2bag(String hS) {

        SortedMap<String,Set<String>> res = new TreeMap<String,Set<String>>();
        String[] els = hS.split("\\|");

        for (String s : els) {
            String[] h_p = s.split(",");
            Set<String> images = new HashSet<String>();
            for (int i=1;i<h_p.length;i++) {
                images.add(h_p[i]);
            }
            res.put(h_p[0],images);
        }
       
        return res;
    }
    //---------------------------------------------------------------------------
    //"hB" is of the form hB[p4]={b1,b2},hB[p1]={b4},...
    // returns a string of the form "p1,b4|p5,b1|p2,b3|p3,b2|p4,b1,b2"
    public static String h2string(SortedMap<String,Set<String>> hB) {
        String res = "";

        for (SortedMap.Entry<String,Set<String>> entry : hB.entrySet()) {
            res = res + entry.getKey();
            for (String s : entry.getValue()) {
                res = res + "," + s;
            }
            res = res + "|";
        }

        if (res.endsWith("|")) {
            res = res.substring(0, res.length() - 1);
        }
       
        return res;
    }
    //---------------------------------------------------------------------------
    // Efficiency must be improved!!!!
    // Pre: f1="a,!b,c",f2="d,b,c",and neither f1 nor f2 have contradictions
    // Post: Can we find some "x" and "!x"?
    public static boolean isThereAContradiction(String f1,String f2) {
        String[] sF1 = f1.split(",");
        String[] sF2 = f2.split(",");
        for (String s1 : sF1) {
            for(String s2 : sF2) {
                // System.out.println(s1 + "-" + s2);
                if (s1.startsWith("!") && (s2.equals(s1.substring(1)))) {
                    return true;
                }
                if (s2.startsWith("!") && (s1.equals(s2.substring(1)))) {
                    return true;
                }
            }
        }
        return false;
    }

    // "FormPresAndPosts" is a number of triples of the form "String robotForm,String PreCap,String PostCap"
    public static boolean gef_many(String buchForm,String... FormPresAndPosts) {
        SMultiSet preMS = new SMultiSet();
        SMultiSet postMS = new SMultiSet();
        String robotForm = "";

        int nRobots = FormPresAndPosts.length / 3;

        int pos = 0;
        for (int r=0;r<nRobots;r++) {
            if (r == 0) {
                robotForm = FormPresAndPosts[pos];
            }
            else {
                robotForm = robotForm + "," + FormPresAndPosts[pos];
            }

            preMS.add(getMultiFromString(FormPresAndPosts[pos+1]));
            postMS.add(getMultiFromString(FormPresAndPosts[pos+2]));
            pos += 3;
        }
        boolean res = gef_one(buchForm,robotForm,preMS,postMS);
        numGefs++;
        if (res) {
            numTrues++;
            System.err.print("#robots: " + nRobots + " (" + numGefs + "," + numTrues + ")         \r");
        }
        // DEB("#robots:     " + nRobots);
        // DEB("\tbuchForm:  " + buchForm);
        // DEB("\trobotForm: " + robotForm);
        // DEB("\t#trues:    " + numTrues);
        // DEB("\tgef:       " + res);
        return res;
    }

    // "buchForm" is of the form "b1,b2,!b3,!b4"
    // "robotForm" is of the form "b1,b5" (no negations): after execution, b1&b5
    // "capChange" is a P-multiset establishing the "partitionOcc" variation: neg components
    // represent cap resources engaged, positive values resources freed
    // Computes gef value for one robot transition
    public static boolean gef_one(String buchForm,String robotForm,SMultiSet preMS,SMultiSet postMS) {
        SMultiSet buchMS = getMultiFromString(buchForm);
        SMultiSet robotMS = getMultiFromString(robotForm);

        // capacity must be greaterOrEqual than the new necessities
        SMultiSet newPartitionOccBag = capBag.add(partitionOccBag,preMS);


        if (!capBag.greaterOrEqual(newPartitionOccBag)) {
            return false;
        }

        // Is buchForm true?
        if (buchForm.equals("1")) {
            return true;
        }
        //since pre is verified, we must also add the freed resources
        newPartitionOccBag.add(postMS);

        // Last part of the algorithm, line 12 ...
        Set<String> tS = buchMS.keySet();
        
        for (String b : B) {
            int rOcc = ROIOccupancy(newPartitionOccBag,b);
            if (tS.contains(b) && (rOcc == 0)) {
                return false;
            }
            if (tS.contains("!"+b) && (rOcc >= 1)) {
                return false;
            }
        }
        
        return true;
    }
    // Renew does not like a variable number of arguments in the functions labelling transitions
    // (the "..." construct in function definition). So, a function must be defined for each
    // transition synchronizing from upto n robots
    // This must be done for "gef" and "updateCap" functions
    public static boolean gef(String buchForm,String robForm1,String PreCap1,String PostCap1) {
        return gef_many(buchForm,robForm1,PreCap1,PostCap1);
    }

    public static boolean gef(String buchForm,
                              String robotForm1,String PreCap1,String PostCap1,
                              String robotForm2,String PreCap2,String PostCap2) {
        return gef_many(buchForm,robotForm1,PreCap1,PostCap1,robotForm2,PreCap2,PostCap2);
    }

    public static boolean gef(String buchForm,
                              String robotForm1,String PreCap1,String PostCap1,
                              String robotForm2,String PreCap2,String PostCap2,
                              String robotForm3,String PreCap3,String PostCap3) {
        return gef_many(buchForm,robotForm1,PreCap1,PostCap1,robotForm2,PreCap2,PostCap2,robotForm3,PreCap3,PostCap3);
    }
    //4
    public static boolean gef(String buchForm,
                              String robotForm1,String PreCap1,String PostCap1,
                              String robotForm2,String PreCap2,String PostCap2,
                              String robotForm3,String PreCap3,String PostCap3,
                              String robotForm4,String PreCap4,String PostCap4) {
        return gef_many(buchForm,robotForm1,PreCap1,PostCap1,robotForm2,PreCap2,PostCap2,
                       robotForm3,PreCap3,PostCap3,robotForm4,PreCap4,PostCap4);
    }
    //5
    public static boolean gef(String buchForm,
                              String robotForm1,String PreCap1,String PostCap1,
                              String robotForm2,String PreCap2,String PostCap2,
                              String robotForm3,String PreCap3,String PostCap3,
                              String robotForm4,String PreCap4,String PostCap4,
                              String robotForm5,String PreCap5,String PostCap5) {
        return gef_many(buchForm,robotForm1,PreCap1,PostCap1,robotForm2,PreCap2,PostCap2,
                       robotForm3,PreCap3,PostCap3,robotForm4,PreCap4,PostCap4,
                       robotForm5,PreCap5,PostCap5
        );
    }
    //6
    public static boolean gef(String buchForm,
                              String robotForm1,String PreCap1,String PostCap1,
                              String robotForm2,String PreCap2,String PostCap2,
                              String robotForm3,String PreCap3,String PostCap3,
                              String robotForm4,String PreCap4,String PostCap4,
                              String robotForm5,String PreCap5,String PostCap5,
                              String robotForm6,String PreCap6,String PostCap6) {
        return gef_many(buchForm,robotForm1,PreCap1,PostCap1,robotForm2,PreCap2,PostCap2,
                       robotForm3,PreCap3,PostCap3,robotForm4,PreCap4,PostCap4,
                       robotForm5,PreCap5,PostCap5,robotForm6,PreCap6,PostCap6
        );
    }
    //7
    public static boolean gef(String buchForm,
                              String robotForm1,String PreCap1,String PostCap1,
                              String robotForm2,String PreCap2,String PostCap2,
                              String robotForm3,String PreCap3,String PostCap3,
                              String robotForm4,String PreCap4,String PostCap4,
                              String robotForm5,String PreCap5,String PostCap5,
                              String robotForm6,String PreCap6,String PostCap6,
                              String robotForm7,String PreCap7,String PostCap7) {
        return gef_many(buchForm,robotForm1,PreCap1,PostCap1,robotForm2,PreCap2,PostCap2,
                       robotForm3,PreCap3,PostCap3,robotForm4,PreCap4,PostCap4,
                       robotForm5,PreCap5,PostCap5,robotForm6,PreCap6,PostCap6,
                       robotForm7,PreCap7,PostCap7
        );
    }
    //8
    public static boolean gef(String buchForm,
                              String robotForm1,String PreCap1,String PostCap1,
                              String robotForm2,String PreCap2,String PostCap2,
                              String robotForm3,String PreCap3,String PostCap3,
                              String robotForm4,String PreCap4,String PostCap4,
                              String robotForm5,String PreCap5,String PostCap5,
                              String robotForm6,String PreCap6,String PostCap6,
                              String robotForm7,String PreCap7,String PostCap7,
                              String robotForm8,String PreCap8,String PostCap8) {
        return gef_many(buchForm,robotForm1,PreCap1,PostCap1,robotForm2,PreCap2,PostCap2,
                       robotForm3,PreCap3,PostCap3,robotForm4,PreCap4,PostCap4,
                       robotForm5,PreCap5,PostCap5,robotForm6,PreCap6,PostCap6,
                       robotForm7,PreCap7,PostCap7,robotForm8,PreCap8,PostCap8
        );
    }
    //9
    public static boolean gef(String buchForm,
                              String robotForm1,String PreCap1,String PostCap1,
                              String robotForm2,String PreCap2,String PostCap2,
                              String robotForm3,String PreCap3,String PostCap3,
                              String robotForm4,String PreCap4,String PostCap4,
                              String robotForm5,String PreCap5,String PostCap5,
                              String robotForm6,String PreCap6,String PostCap6,
                              String robotForm7,String PreCap7,String PostCap7,
                              String robotForm8,String PreCap8,String PostCap8,
                              String robotForm9,String PreCap9,String PostCap9) {
        return gef_many(buchForm,robotForm1,PreCap1,PostCap1,robotForm2,PreCap2,PostCap2,
                       robotForm3,PreCap3,PostCap3,robotForm4,PreCap4,PostCap4,
                       robotForm5,PreCap5,PostCap5,robotForm6,PreCap6,PostCap6,
                       robotForm7,PreCap7,PostCap7,robotForm8,PreCap8,PostCap8,
                       robotForm9,PreCap9,PostCap9
        );
    }
    //10
    public static boolean gef(String buchForm,
                              String robotForm1,String PreCap1,String PostCap1,
                              String robotForm2,String PreCap2,String PostCap2,
                              String robotForm3,String PreCap3,String PostCap3,
                              String robotForm4,String PreCap4,String PostCap4,
                              String robotForm5,String PreCap5,String PostCap5,
                              String robotForm6,String PreCap6,String PostCap6,
                              String robotForm7,String PreCap7,String PostCap7,
                              String robotForm8,String PreCap8,String PostCap8,
                              String robotForm9,String PreCap9,String PostCap9,
                              String robotForm10,String PreCap10,String PostCap10) {
        return gef_many(buchForm,robotForm1,PreCap1,PostCap1,robotForm2,PreCap2,PostCap2,
                       robotForm3,PreCap3,PostCap3,robotForm4,PreCap4,PostCap4,
                       robotForm5,PreCap5,PostCap5,robotForm6,PreCap6,PostCap6,
                       robotForm7,PreCap7,PostCap7,robotForm8,PreCap8,PostCap8,
                       robotForm9,PreCap9,PostCap9,robotForm10,PreCap10,PostCap10
        );
    }
    //---------------------------------------------------------------------------
    // "PrePost" is a number of pairs of the form "preMS,postMS" 
    public static void updateCap_many(String... PrePost) {
        int nPairs = PrePost.length / 2;

        int pos = 0;
        for (int p=0;p<nPairs;p++) {
            //[pos]: pre multiset; [pos+1]: post multiset
            partitionOccBag.add(getMultiFromString(PrePost[pos])); //engaging new capacity
            partitionOccBag.diff(getMultiFromString(PrePost[pos+1])); //freeing capacity
            pos += 2;
        }
    }
    
    public static void updateCap(String pre1,String post1) {
        updateCap_many(pre1,post1);
    }

    public static void updateCap(String pre1,String post1,String pre2,String post2) {
        updateCap_many(pre1,post1,pre2,post2);
    }

    public static void updateCap(String pre1,String post1,String pre2,String post2,String pre3,String post3) {
        updateCap_many(pre1,post1,pre2,post2,pre3,post3);
    }

    //generated with generate_eval_cap_code.py
    public static void updateCap(
        String pre1,String post1,
        String pre2,String post2,
        String pre3,String post3,
        String pre4,String post4
    ) {
        updateCap_many(
            pre1,post1,
            pre2,post2,
            pre3,post3,
            pre4,post4
        );
    }

    public static void updateCap(
        String pre1,String post1,
        String pre2,String post2,
        String pre3,String post3,
        String pre4,String post4,
        String pre5,String post5
    ) {
        updateCap_many(
            pre1,post1,
            pre2,post2,
            pre3,post3,
            pre4,post4,
            pre5,post5
        );
    }

    public static void updateCap(
        String pre1,String post1,
        String pre2,String post2,
        String pre3,String post3,
        String pre4,String post4,
        String pre5,String post5,
        String pre6,String post6
    ) {
        updateCap_many(
            pre1,post1,
            pre2,post2,
            pre3,post3,
            pre4,post4,
            pre5,post5,
            pre6,post6
        );
    }

    public static void updateCap(
        String pre1,String post1,
        String pre2,String post2,
        String pre3,String post3,
        String pre4,String post4,
        String pre5,String post5,
        String pre6,String post6,
        String pre7,String post7
    ) {
        updateCap_many(
            pre1,post1,
            pre2,post2,
            pre3,post3,
            pre4,post4,
            pre5,post5,
            pre6,post6,
            pre7,post7
        );
    }

    public static void updateCap(
        String pre1,String post1,
        String pre2,String post2,
        String pre3,String post3,
        String pre4,String post4,
        String pre5,String post5,
        String pre6,String post6,
        String pre7,String post7,
        String pre8,String post8
    ) {
        updateCap_many(
            pre1,post1,
            pre2,post2,
            pre3,post3,
            pre4,post4,
            pre5,post5,
            pre6,post6,
            pre7,post7,
            pre8,post8
        );
    }

    public static void updateCap(
        String pre1,String post1,
        String pre2,String post2,
        String pre3,String post3,
        String pre4,String post4,
        String pre5,String post5,
        String pre6,String post6,
        String pre7,String post7,
        String pre8,String post8,
        String pre9,String post9
    ) {
        updateCap_many(
            pre1,post1,
            pre2,post2,
            pre3,post3,
            pre4,post4,
            pre5,post5,
            pre6,post6,
            pre7,post7,
            pre8,post8,
            pre9,post9
        );
    }

    public static void updateCap(
        String pre1,String post1,
        String pre2,String post2,
        String pre3,String post3,
        String pre4,String post4,
        String pre5,String post5,
        String pre6,String post6,
        String pre7,String post7,
        String pre8,String post8,
        String pre9,String post9,
        String pre10,String post10
    ) {
        updateCap_many(
            pre1,post1,
            pre2,post2,
            pre3,post3,
            pre4,post4,
            pre5,post5,
            pre6,post6,
            pre7,post7,
            pre8,post8,
            pre9,post9,
            pre10,post10
        );
    }

    public static void updateCap(
        String pre1,String post1,
        String pre2,String post2,
        String pre3,String post3,
        String pre4,String post4,
        String pre5,String post5,
        String pre6,String post6,
        String pre7,String post7,
        String pre8,String post8,
        String pre9,String post9,
        String pre10,String post10,
        String pre11,String post11
    ) {
        updateCap_many(
            pre1,post1,
            pre2,post2,
            pre3,post3,
            pre4,post4,
            pre5,post5,
            pre6,post6,
            pre7,post7,
            pre8,post8,
            pre9,post9,
            pre10,post10,
            pre11,post11
        );
    }

    //---------------------------------------------------------------------------
    // "b" is a ROI, "partitionOccBag" is an occupancy multiset
    // returns the marking of places p\in P such that b\in h(p)
    private static int ROIOccupancy(SMultiSet partitionOccBag,String b) {
        int res = 0;
        for (String key : partitionOccBag.keySet()) {
            if (hBag.get(key).contains(b)) {
                res = res + partitionOccBag.getVal(key);
            }
        }

        return res;
    }
    //---------------------------------------------------------------------------
    //For testing
    public static void main(String[] arg) {
        String cap = "1'p11,1'p12,1'p13,4'p14,1'p15,1'p21,1'p22,1'p23,5'p24,1'p25,1'p26,1'p31,1'p32,1'p33,3'p34";
        String partOcc = "4'p14,4'p24,3'p34";
        String h = "p11,d|p12,c|p13,k|p14,w1|p15,a|p21,e|p22,f|p23,g|p24,w2|p25,b|p26,j|p31,h|p32,f|p33,i|p34,w3";
        Eval eval = new Eval(cap,partOcc,h);

        System.out.println("h:" + h2string(h2bag(h)));
        System.out.println("partitionOccBag:" + partitionOccBag.toString());
        System.out.println("d: " + ROIOccupancy(partitionOccBag,"d"));
        System.out.println("w1: " + ROIOccupancy(partitionOccBag,"w1"));

        // move from p14 to p11
        if(gef("!c","d","1'p11","1'p14")) {
            System.out.println("OK");
            updateCap("1'p11","1'p14");
            System.out.println("new occ:" + partitionOccBag.toString());
        }
        else {
            System.out.println("A bad situation");
        }

        //System.out.println(isThereAContradiction("a,b","a,c"));
        //System.out.println(isThereAContradiction("a,b","c,!b"));

        //System.out.println(getMultiFromString("1'a,2'c").toString());
        
    }
    // for debugging purposes
    static private void DEB(String mens) {
        System.err.println(mens);
    }
}
