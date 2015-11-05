/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package binarytreesom.clustering;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import weka.core.Instances;
import weka.gui.beans.Clusterer;

/**
 *
 * @author César Astudillo
 */
public class BinaryTreeSOMClustering extends Clusterer {
    
    /**
     * Hierarchical Clustering Visualization in R
     * https://cran.r-project.org/web/packages/dendextend/vignettes/Cluster_Analysis.html
     */
    
    /**
     * The weight vector.
     */
    private double weight[][];
    
    
    
    
    /**
     * The Instance currently being examined.
     */
    private double x[];
    /**
     * The height of the tree.
     */
    private final int heightOfTheTree;
    
    /**
     * The learning rate.
     */
    private double currentAlpha;
    
    /**
     * the number of neurons.
     */
    private final int numberOfNeurons;
    
    
    //private int i,j; //miscelaneous counters

    private int dimensionality;
    private double [][] data;
    private int numberOfInstances;
    
    /**
     * The maximum number of iterations.
     */
    private final int maximumNumberofIterations;

    private final double initialAlpha;
    private final double finalAlpha;
    private final long seed;
    private final String filenameARFF;
    private final Random r;
    
    
    /**
     * constructor
     * @param filenameARFF the name of the ARFF file to be open.
     * @param maximumNumerofIterations The maximum number of iterations.
     * @param initialAlpha The initial learning rate.
     * @param finalAlpha The final learning rate.
     * @param heightOfTheTree The height of the tree.
     * @param seed The pseudo random seed.
     * @throws java.io.IOException
     */
    public BinaryTreeSOMClustering(String filenameARFF, int maximumNumerofIterations, double initialAlpha, double finalAlpha, int heightOfTheTree, int seed) throws IOException{
    this.filenameARFF=filenameARFF;
    this.maximumNumberofIterations=maximumNumerofIterations;
    this.currentAlpha=initialAlpha;
    this.initialAlpha=initialAlpha;
    this.finalAlpha=finalAlpha;
    this.heightOfTheTree=heightOfTheTree;//height of the tree 
    this.seed=seed;
    r=new Random(seed);
    this.numberOfNeurons=(int)Math.pow(2, getHeightOfTheTree())-1;//number of neurons
    initialize();
    }
    
/**
 * Initialize the tree configuration. This implementation considers a complete binary tree of depth h. 
 */    
private void initialize() throws IOException{
   //the number of nodes N, is penednt on h. actualy N
   //h=ln N -> N=2^h
    Instances instances = readArff(getFilenameARFF());
    instances.setClassIndex(-1);//clustering Stuff
    
    numberOfInstances=instances.numInstances();
    dimensionality=instances.numAttributes();
    data = new double[getNumberOfInstances()][getDimensionality()];
    weight = new double[getNumberOfNeurons()][getDimensionality()];
    //randomly select instances and assign to weight.
    
    for(int k = 0; k < getNumberOfNeurons(); k++) {
            weight[k]=instances.instance(r.nextInt(getNumberOfInstances())).toDoubleArray(); //hard copy of the double array
        }
    
    for(int k = 0; k < getNumberOfInstances(); k++) {
            data[k]=instances.instance(k).toDoubleArray(); //hard copy of the double array
        }
}  
    
  /**
  * SOM's update rule.
  * @param alpha learning rate
  * @param x current instance
  * @param w weight vector
  */
    public static void update(final double alpha, final double []x, double []w){
        for(int i = 0; i < w.length; i++) {
                //if(x[i]!=Double.NaN) //if the number exists
            if(!Double.isNaN(x[i])) //if the number exists
                    w[i]=w[i]*(1 - alpha)+alpha*x[i];
            }
    }
    

    /**
     * Heuristic method for finding the best matching Unit (BMU). The method iteratively traverse the binary search tree choosing the child which is closer
     * to the instance currently being examined.
     * The method starts the search from the root and always reach one of the leaves.
     * Each time a node is identified as being closer when competing with his sibling, it is updated.
     * The searching process continues down to the leaves through the path that includes the node currently identified as the closest.
     * The process is repeated until a leaf is reached.
     * @param x  the input vector
     * @return  Index to the neuron identified as the best matching unit BMU.
     */
    public int findBest(final double []x){
        // http://stackoverflow.com/questions/8256222/binary-tree-represented-using-array
        
        int N=0;//root
        update(currentAlpha,x,weight[N]);//always update root
        
        
        
        while (2*N+2< numberOfNeurons) {
            N=2*N;
            //System.out.println("d x,"+(N+1)+" = "+getDistance(x,getWeight()[N+1]));
            //System.out.println("d x,"+(N+2)+" = "+getDistance(x,getWeight()[N+2]));
            
            if (   getDistance(x,weight[N+1])  <   getDistance(x,weight[N+2])   ){
                
                N=N+1;
                update(currentAlpha,x,weight[N]);//update left subtree
            }
            else{
                N=N+2;
                update(currentAlpha,x,weight[N]);//update right subtree
            }
                
        }
        return N;
 }
    
/**
 * Dissimilarity function. currently Euclidean distance.
 * @param x The input vector.
 * @param w The weight vector.
 * @return The distance between the input and weight vector.
 */
    private static double getDistance(final double[] x, final double[] w) {
        double distance;
        distance = 0.0;
        //euclidean distance
        for (int k = 0; k < x.length; k++) {
            if(!Double.isNaN(x[k])) //if the number exists
                distance+=(x[k]-w[k])*(x[k]-w[k]);
        }
        return Math.sqrt(distance);
    }
    
    
   /**
     * Loading data from a given ARFF file which contains the data set.
     * @param filename path to the file which contains the data set
     * @return data set into Instances
     * @throws java.io.IOException
     */
    public static Instances readArff(String filename) throws IOException{
        Instances data;
            BufferedReader reader = new BufferedReader(new FileReader(filename));
            data = new Instances(reader);
            reader.close();
            //data.setClassIndex(data.numAttributes() - 1);
            return data;
    }
    
    /**
     * The learning loop. It systematically select input at random and train the network.
     */
public void learn(){
    for (int k = 0; k < getMaximumNumberofIterations(); k++) {
        x=  getData()[r.nextInt(numberOfInstances)];//randomly select an instance
        findBest(x);
        schedule();
        //System.out.println(""+k);
    }
}    

/**
 * updates the parameter values according to a pre-defined decaying schedule.
 */
    private void schedule() {
        currentAlpha=computeCurrentParameterValue(currentAlpha, initialAlpha, finalAlpha, maximumNumberofIterations);
    }
   
    
    
    /**
     * Computes the current value of a parameter (for example, 
     * learning rate or radius).
     * @param currentValue current value of the parameter.
     * @param initialValue initial value of the parameter.
     * @param finalValue final value of the parameter.
     * @param maximumNumberofIterations The maximum number of iterations.
     * @return the new value of the parameter. 
     */
    public static double computeCurrentParameterValue(final double currentValue, final double initialValue, 
            final double finalValue,
            final int maximumNumberofIterations){
 
        double delta = (initialValue - finalValue)/maximumNumberofIterations;
        //currentValue -= delta;
        return currentValue - delta;
     }

    /**
     * @return the weight vectors
     */
    public double[][] getWeight() {
        return weight;
    }

    /**
     * @return the height of the tree
     */
    public int getHeightOfTheTree() {
        return heightOfTheTree;
    }

    /**
     * @return the current learning rate
     */
    public double getCurrentAlpha() {
        return currentAlpha;
    }

    /**
     * @return the number of neurons
     */
    public int getNumberOfNeurons() {
        return numberOfNeurons;
    }

    /**
     * @return the maximum number of iterations
     */
    public int getMaximumNumberofIterations() {
        return maximumNumberofIterations;
    }

    /**
     * @return the dimensionality
     */
    public int getDimensionality() {
        return dimensionality;
    }

    /**
     * @return the initial learning rate
     */
    public double getInitialAlpha() {
        return initialAlpha;
    }

    /**
     * @return the final learning rate
     */
    public double getFinalAlpha() {
        return finalAlpha;
    }

    /**
     * @return the data
     */
    public double[][] getData() {
        return data;
    }

    /**
     * @return the seed
     */
    public long getSeed() {
        return seed;
    }

    /**
     * @return the filenameARFF
     */
    public String getFilenameARFF() {
        return filenameARFF;
    }

    /**
     * @return the numberOfInstances
     */
    public int getNumberOfInstances() {
        return numberOfInstances;
    }
    
}


