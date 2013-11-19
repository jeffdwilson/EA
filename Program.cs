using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace EA_j2
{
    // Global settings and variables
    public class G
    {
        // evolution parameters
        public static readonly string geneType = "double";                      // options are "binary", "double"
        public static readonly int[] netDef = { 2,2,1 };                        // not used for non neural net evolution
        public static readonly int geneLen = FFNN.NumberOfWeights( netDef );    // length of gene, set to FFNN.NumberOfWeights( netDef ) for neural nets

        // population size and turnover
        public static readonly int popSize = 30;                        // population size
        public static readonly double elitism = 0.25;                   // fraction that live on to next generation
        public static readonly int litterSize = 3;                      // number of offspring created from parent or parents

        // mating
        public static readonly double asexual = 0.5;                    // fraction of litters that will be born from single parent
        public static readonly double bestToMate = 0.5;                // of those chosen to mate, fraction chosen because they are best
        public static readonly string mateSelectMethod = "map";         // options are best-worst, random, roulette, rank, map
        public static readonly double mateBWratio = 100;                // odds of best fitness mating compared to worst
        public static readonly string pairingMethod = "random";         // options are "random"

        // reproduction
        public static readonly string crossOverMethod = "random";       // options are "one-point", "two-point", "random"
        public static readonly double crossOverRate = 0.10;             // odds of "random" cross over occuring
        public static readonly double mutationRate = 0.05;             // odds of any value in gene changing
        public static readonly double mutationFactor = 2.0;             // geneType "double" multiply or divide by 1.0 to [this value]

        // death selection
        public static readonly double protectFromDeath = 1.0;           // of elite creatures, fraction protected by being best; 1.0->worst die, 0.5->half top half selected
        public static readonly string deathSelectMethod = "map";        // options are best-worst, random, roulette, rank, map 
        public static readonly double deathBWratio = 10;                // odds of best fitness dying compared to worst

    }

    // Program Class
    class Program
    {
        // fields
        static Random random = new Random(  );

        static void Main( string[] args ) {

            // any setup

            // run evolution
            Console.WriteLine( "working...\n" );

            bool success = EA.Run(
                getRandomGeneFunc: GetRandomGene,
                getPopFitnessFunc: GetPopFitness );

            // show result
            Console.WriteLine( "success = {0}", success );
            Console.ReadKey( false );

        }

        // functions for hooking EA to the problem at hand
        static double[] GetRandomGene() {
            //return GetRandGene_HelloWorld( G.geneLen );
            return FFNN.RandomWeights( G.netDef );
        }
        static bool GetPopFitness( double[][] genes, ref double[] popFitness, out double bestFitness, out double avrFitness ) {
            //return Fitness_HelloWorld( genes, ref popFitness, out bestFitness, out avrFitness );
            return NE1.EvaluateXOR( genes, ref popFitness, out bestFitness, out avrFitness );
        }

        // functions for EA hello world
        /* public static double[] GetRandGene_HelloWorld( int geneLen ) {
            var geneOut = new double[geneLen];
            for( int i = 0; i < geneLen; i++ )
                geneOut[i] = random.NextDouble() <= 0.5 ? 0.0 : 1.0;
            return geneOut;
        }
        public static bool Fitness_HelloWorld( double[][] genes, ref double[] popFitness, out double bestFitness, out double avrFitness ) {
            // EA hello world, count number of 1's in gene

            int popSize = genes.Length;
            int geneLen = genes[0].Length;
            double thisFitness;
            bestFitness = 0;
            avrFitness = 0;

            // loop thru population
            for( int i = 0; i < popSize; i++ ) {
                thisFitness = popFitness[i];

                // calculate fitness if it's now -1
                if( thisFitness == -1.0 ) {
                    thisFitness = 0;
                    for( int j = 0; j < geneLen; j++ )
                        if( genes[i][j] == 1.0 ) thisFitness++;
                    popFitness[i] = thisFitness;
                }

                avrFitness += thisFitness;
                bestFitness = Math.Max( bestFitness, thisFitness );

            }
            avrFitness /= popSize;

            return bestFitness == geneLen;
        } */
    }

    // Neural Evolution class -1
    public class NE1
    {
        // fields
        static FFNN nn = new FFNN( G.netDef );

        public static bool EvaluateXOR( double[][] genes, ref double[] popFitness, out double bestFitness, out double avrFitness ) {

            double[] inputValues = {0.0, 0.1, 0.2, 0.3, 0.7 ,0.8, 0.9, 1.0};
            double[] nnInputs = new double[2];
            double[] nnOutputs = new double[1];
            double nnOutput, correctOutput, thisFitness;
            
            bestFitness = Double.MinValue;
            avrFitness = 0.0; thisFitness = 0;
            
            // loop over genes
            for( int i=0; i<genes.Length; i++){

                // load nn with gene
                nn.Weights = genes[i];
                thisFitness = 0;

                // loop through input set
                foreach( double A in inputValues){
                    foreach( double B in inputValues){

                        // evaluate nn for one input
                        nnInputs[0] = A;
                        nnInputs[1] = B;
                        nnOutputs = nn.Output( nnInputs );
                        nnOutput = nnOutputs[0];

                        // correct answer
                        correctOutput = Math.Round(A) != Math.Round(B)? 1.0 : 0.0;  // XOR

                        // score by distance from 0.5 with big penalty for being wrong
                        if( Math.Round(nnOutput) != correctOutput )
                            thisFitness += (-100 - Math.Abs( 0.5 - nnOutput ));
                        else
                            thisFitness += (Math.Abs( 0.5 - nnOutput ));
                    }
                }
                
                // summary
                popFitness[i] = thisFitness;
                bestFitness = Math.Max( bestFitness, thisFitness );
                avrFitness += thisFitness;
            }

            // goal achieved (perfect score is 32)
            return thisFitness >= 31.99999;
        }
    }

    // EA Class
    public class EA
    {
        // Fields
        private static Population myPop;
        private static int displayMode = 0;
        private static DateTime nextProgress = DateTime.Now;

        // MAIN FUNCTION
        public static bool Run( RandomGeneSig getRandomGeneFunc, FitnessSig getPopFitnessFunc ) {
            // Runs the EA algorythm, returns true if goal is achieved, false if time is exceeded

            // local variables
            bool goalAchieved;
            double bestFitness, avrFitness;
            double[] popFitness = new double[G.popSize];
            double[][] genes;

            // Create a random population of creatures
            myPop = new Population( G.popSize, G.geneLen, getRandomGeneFunc );

            // MAIN LOOP
            while( true ) {

                // Evaluate and sort population
                genes = myPop.GetGenes();
                myPop.GetFitness( ref popFitness );
                goalAchieved = getPopFitnessFunc( genes, ref popFitness, out bestFitness, out avrFitness );
                myPop.SetFitness( popFitness, bestFitness, avrFitness );
                myPop.Sort();
                //myPop.Print( "EVALUATED", false );
                //Console.ReadKey( true );

                // Are we done?
                if( goalAchieved ) break;

                // Mate Selection
                myPop.SelectToMate( showStepsFlag: true );
                myPop.MatePairing();

                // Create New Creatures
                myPop.CreateOffspring();
                myPop.Mutate();

                // Death Selection
                myPop.SelectToDie( showStepsFlag: false );

                // Pause or continue
                PausePrintContinue();

                // Update population (genes)
                myPop.InsertNewOffspring();
                myPop.IncGeneration();
            }

            // success
            myPop.Print( "(SUCCESS)", false );
            FFNN answer = new FFNN( G.netDef, myPop.creatures[0].Gene);
            Console.WriteLine( answer.ToString() );

            return goalAchieved;
        }

        private static void PausePrintContinue() {

            int fastModeSeconds = 1;
            ConsoleKeyInfo ki;
            char? keyPressed;

            while( true ) {

                //get last space or enter already pressed, ignore others
                keyPressed = null;
                while( Console.KeyAvailable ) {
                    ki = Console.ReadKey( true );
                    if( ki.KeyChar == ' ' || ki.KeyChar == 13 )
                        keyPressed = ki.KeyChar;
                }

                //state machine
                // displayMode: 0=print and wait, 1=wait for space, 2=show while running, 3=fast mode
                switch( displayMode ) {

                    case 0: //print and wait

                        myPop.Print( "(STEP MODE)", true );
                        displayMode = 1; //wait
                        break;

                    case 1: // wait

                        if( keyPressed == ' ' ) {
                            displayMode = 0; //print and wait
                            return;
                        } else if( keyPressed == 13 ) {
                            displayMode = 2; // show while running
                            return;
                        }
                        System.Threading.Thread.Sleep( 100 );
                        break;

                    case 2: // show while running

                        if( keyPressed == ' ' ) {
                            displayMode = 0; //print and wait
                        } else if( keyPressed == 13 ) {
                            displayMode = 3; //fast
                            nextProgress = DateTime.Now;
                        } else {
                            myPop.Print( "(SHOW MODE)", true );
                            System.Threading.Thread.Sleep( 500 );
                            return;
                        }
                        break;

                    case 3: // fast mode

                        if( keyPressed == ' ' ) {
                            displayMode = 0; //print and wait
                        } else if( keyPressed == 13 ) {
                            displayMode = 2;
                        } else if( DateTime.Now >= nextProgress ) {
                            nextProgress = DateTime.Now.AddSeconds( fastModeSeconds );
                            myPop.Print( "(FAST MODE)", true );
                            return;
                        } else {
                            return;
                        }
                        break;

                    default:
                        throw new ArgumentException( "displayMode out of range = " + displayMode );
                }
            }
        }

        // Population class
        public class Population
        {
            public readonly int popSize;
            public readonly int geneLen;

            private int generation, oldestAge, ageOfBest;
            private double bestFitness, avrFitness;

            public Creature[] creatures;
            private List<int> toMate;
            private List<IntPair> matingPairs;
            private List<Creature> offspring;
            private List<int> toDie;

            private readonly int elite, protectFromDeathI, turnOver, numLitters, numSexual, numAsexual, numToMate;
            private static Int32 seed = 0;

            // Construtor
            public Population( int popSize, int geneLen, RandomGeneSig getRandomGeneFunc ) {

                // init simple fields
                this.popSize = popSize;
                this.geneLen = geneLen;

                // create creatures with random genes
                creatures = new Creature[popSize];
                for( int i = 0; i < popSize; i++ )
                    creatures[i] = new Creature( getRandomGeneFunc() );

                // create blank supporting lists
                toMate = new List<int>( popSize );
                matingPairs = new List<IntPair>( popSize );
                offspring = new List<Creature>( popSize );
                toDie = new List<int>( popSize );

                // calculate some derived constants
                elite = (int)( popSize * G.elitism );                                 // number of creatures that will live on
                protectFromDeathI = (int)Math.Round( elite * G.protectFromDeath );  // number of best creatures protected from death
                turnOver = popSize - elite;                                         // number of creatures to kill and recreate
                numLitters = (int)Math.Ceiling( turnOver / (double)G.litterSize );  // number of reqired litters
                numAsexual = (int)Math.Round( numLitters * G.asexual );             // number of asexual litters
                numSexual = numLitters - numAsexual;                                // number of sexual litters
                numToMate = numAsexual + 2 * numSexual;                             // number of creature to select for mating (not all may be used)
                if( numToMate > popSize ) throw new ArgumentException( "litter size to small to maintain population" );

            }

            // Get genes method
            public double[][] GetGenes() {
                // returns gene array references
                double[][] genesOut = new double[popSize][];
                for( int i = 0; i < popSize; i++ )
                    genesOut[i] = creatures[i].Gene;
                return genesOut;
            }

            // Get fitness method
            public void GetFitness( ref double[] popFitness ) {
                for( int i = 0; i < popSize; i++ )
                    popFitness[i] = creatures[i].Fitness;
            }

            // Set fitness method
            public void SetFitness( double[] popFitness, double bestFitness, double avrFitness ) {
                this.bestFitness = bestFitness;
                this.avrFitness = avrFitness;
                for( int i = 0; i < popSize; i++ )
                    creatures[i].Fitness = popFitness[i];
            }

            // Sort method (and set ageOfBest)
            public void Sort() {
                Array.Sort( creatures, Creature.FitnessDesending );
                // now any indexes in the supporting index lists are corrupt
                toDie.Clear();
                toMate.Clear();
                ageOfBest = creatures[0].Age;
            }

            // Increment generation and ages
            public void IncGeneration() {
                generation++;
                oldestAge = 0;
                for( int i = 0; i < popSize; i++ )
                    oldestAge = Math.Max( oldestAge, ++creatures[i].Age );
            }

            // Select to mate, by fitness only
            public void SelectToMate( bool showStepsFlag = false ) {

                // int maxToMate is a field of population
                int bestThatMateI = (int)Math.Ceiling( numToMate * G.bestToMate );
                int toSelect = numToMate - bestThatMateI;

                // result is that toMate is loaded with creature indexes that can mate
                toMate.Clear();

                // first just take the best few because they are the best
                for( int i = 0; i < bestThatMateI; i++ ) toMate.Add( i );

                // create and load selectFromList
                List<SelectionByFitness.SelectFromItem> selectFromList = new List<SelectionByFitness.SelectFromItem>( popSize );
                for( int i = bestThatMateI; i < popSize; i++ )
                    selectFromList.Add( new SelectionByFitness.SelectFromItem( i, creatures[i].Fitness ) );

                // call SelectionByFitness.Select to select additional creatures to mate 
                SelectionByFitness.Select( ref toMate, selectFromList, toSelect, G.mateSelectMethod, G.mateBWratio, true, showStepsFlag );

            }

            // Mate pairing
            public void MatePairing() {

                // random mate pairing is the only method so far
                if( G.pairingMethod != "random" )
                    throw new ArgumentException( "bad mate pairing method, " + G.pairingMethod );

                // variables
                Random random = new Random( seed++ );
                int r1, r2, index1, index2;
                matingPairs.Clear();

                // make copy of toMate so not to disturb toMate for printing
                List<int> toMateCopy = new List<int>( toMate.Count );
                for( int i = 0; i < toMate.Count; i++ ) toMateCopy.Add( toMate[i] );

                // add asexual creature index to list
                for( int i = 0; i < numAsexual; i++ ) {
                    r1 = random.Next( toMateCopy.Count );
                    index1 = toMateCopy[r1];
                    toMateCopy.RemoveAt( r1 );
                    matingPairs.Add( new IntPair( index1, -1 ) );
                }

                // do the pairing
                for( int i = 0; i < numSexual; i++ ) {
                    r1 = random.Next( toMateCopy.Count );
                    index1 = toMateCopy[r1];
                    toMateCopy.RemoveAt( r1 );
                    r2 = random.Next( toMateCopy.Count );
                    index2 = toMateCopy[r2];
                    toMateCopy.RemoveAt( r2 );
                    matingPairs.Add( new IntPair( index1, index2 ) );
                }
            }

            // Create offspring
            public void CreateOffspring() {

                // result is offspring list contains new creatures
                offspring.Clear();
                double[] newGene;
                int[] newGeneSrc;
                bool[] newGeneMut;
                Random random = new Random( seed++ );
                int r1, r2;
                bool fromMom;

                // loop thru mating pairs
                for( int i = 0; i < matingPairs.Count; i++ ) {

                    // asexual mating
                    if( matingPairs[i].B == -1 ) {

                        // for each littermate that will be added to offspring
                        for( int j = 0; j < G.litterSize; j++ ) {
                            if( offspring.Count < turnOver ) {

                                newGene = Common.NewCopy( creatures[matingPairs[i].A].Gene );
                                newGeneSrc = new int[geneLen];
                                newGeneMut = new bool[geneLen];
                                offspring.Add( new Creature( newGene, newGeneSrc, newGeneMut ) );
                            }
                        }

                        // sexual mating
                    } else {

                        // for each littermate that will be added to offspring
                        for( int j = 0; j < G.litterSize; j++ ) {
                            if( offspring.Count < turnOver ) {

                                newGene = new double[geneLen];
                                newGeneSrc = new int[geneLen];
                                newGeneMut = new bool[geneLen];

                                switch( G.crossOverMethod ) {

                                    case "one-point":

                                        r1 = random.Next( geneLen + 1 );  //number of genes from mom, rest from dad
                                        for( int k = 0; k < r1; k++ ) { //mom
                                            newGene[k] = creatures[matingPairs[i].A].Gene[k];
                                            newGeneSrc[k] = 1;
                                        }
                                        for( int k = r1; k < geneLen; k++ ) { //dad
                                            newGene[k] = creatures[matingPairs[i].B].Gene[k];
                                            newGeneSrc[k] = 2;
                                        }
                                        break;

                                    case "two-point":

                                        r1 = random.Next( geneLen + 1 );                  //genes from mom
                                        r2 = random.Next( geneLen + 1 );                  //genes from mom and dad, rest from mom
                                        if( r1 > r2 ) Common.Swap( ref r1, ref r2 );

                                        for( int k = 0; k < r1; k++ ) { //mom
                                            newGene[k] = creatures[matingPairs[i].A].Gene[k];
                                            newGeneSrc[k] = 1;
                                        }
                                        for( int k = r1; k < r2; k++ ) { //dad
                                            newGene[k] = creatures[matingPairs[i].B].Gene[k];
                                            newGeneSrc[k] = 2;
                                        }
                                        for( int k = r2; k < geneLen; k++ ) { //mom
                                            newGene[k] = creatures[matingPairs[i].A].Gene[k];
                                            newGeneSrc[k] = 1;
                                        }
                                        break;

                                    case "random":

                                        fromMom = true;
                                        for( int k = 0; k < geneLen; k++ ) {

                                            if( random.NextDouble() <= G.crossOverRate ) fromMom = !fromMom;

                                            if( fromMom ) {
                                                newGene[k] = creatures[matingPairs[i].A].Gene[k];
                                                newGeneSrc[k] = 1;
                                            } else {
                                                newGene[k] = creatures[matingPairs[i].A].Gene[k];
                                                newGeneSrc[k] = 2;
                                            }
                                        }
                                        break;

                                    default:
                                        throw new ArgumentException( "bad cross method = " + G.crossOverMethod );
                                }

                                offspring.Add( new Creature( newGene, newGeneSrc, newGeneMut ) );
                            }
                        }
                    }
                }
            }

            // Mutate offspring
            public void Mutate() {
                Random random = new Random( seed++ );
                double factor;
                switch( G.geneType ) {
                    case "double":
                        for( int i = 0; i < offspring.Count; i++ )
                            for( int j = 0; j < geneLen; j++ )
                                if( random.NextDouble() <= G.mutationRate ) {
                                    factor = 1.0 + random.NextDouble() * ( G.mutationFactor - 1.0 );
                                    if( random.NextDouble() <= 0.5 ) factor = 1 / factor;
                                    offspring[i].Gene[j] *= factor;
                                    offspring[i].GeneMut[j] = true;
                                }
                        break;
                    case "binary":
                        for( int i = 0; i < offspring.Count; i++ )
                            for( int j = 0; j < geneLen; j++ )
                                if( random.NextDouble() <= G.mutationRate ) {
                                    offspring[i].Gene[j] = ( offspring[i].Gene[j] == 1.0 ? 0.0 : 1.0 );
                                    offspring[i].GeneMut[j] = true;
                                }
                        break;
                    default:
                        throw new ArgumentException( "bad geneType = " + G.geneType );
                }
            }

            // Select to die
            public void SelectToDie( bool showStepsFlag = false ) {

                // result is that toDie is loaded with creature indexes that will die
                toDie.Clear();

                // create and load selectFromList
                List<SelectionByFitness.SelectFromItem> selectFromList = new List<SelectionByFitness.SelectFromItem>( popSize );
                for( int i = protectFromDeathI; i < popSize; i++ )
                    selectFromList.Add( new SelectionByFitness.SelectFromItem( i, creatures[i].Fitness ) );

                // call SelectionByFitness.Select to select additional creatures to mate 
                SelectionByFitness.Select( ref toDie, selectFromList, turnOver, G.deathSelectMethod, G.deathBWratio, false, showStepsFlag );

            }

            // Insert offspring over dead creatures
            public void InsertNewOffspring() {
                for( int i = 0; i < toDie.Count; i++ ) {
                    creatures[toDie[i]].Assign( offspring.Last() );
                    offspring.RemoveAt( offspring.Count - 1 );
                }
            }

            // Print population
            public void Print( string message, bool nextStepsFlag = false ) {

                // print colors
                var colors = new Dictionary<string, ConsoleColor>( 5 );
                colors.Add( "defForeground", ConsoleColor.Gray );
                colors.Add( "defBackground", ConsoleColor.Black );
                colors.Add( "mutForeground", ConsoleColor.Red );
                colors.Add( "brightForeground", ConsoleColor.White );
                colors.Add( "momBackground", ConsoleColor.Green );
                colors.Add( "dadBackground", ConsoleColor.Blue );

                // set to default
                Console.ForegroundColor = colors["defForeground"];
                Console.BackgroundColor = colors["defBackground"];

                // print heading
                string line = new String( '=', 78 );
                Console.WriteLine( "\n" + line );
                Console.WriteLine( " Generation = {0}, BEST FITNESS = {1},  {2}", generation, Common.FrmDbl( bestFitness, 6, "zeros-right" ), message );
                Console.WriteLine( "   Age of best = {0}, Age of oldest = {1}, Avr fitness = {2}", ageOfBest, oldestAge, Common.FrmDbl( avrFitness, 6, "zeros-right" ) );
                Console.WriteLine( line );

                // print each creature
                string note;
                for( int i = 0; i < popSize; i++ ) {
                    note = ( toMate.Contains( i ) ? "-mate  " : "       " )
                        + ( toDie.Contains( i ) ? "-die" : "    " );
                    creatures[i].Print( i, note, false, colors );
                    //if( G.geneType != "binary" ) Console.WriteLine( "" );
                }
                Console.WriteLine( "" );

                // print mating pairs and offspring
                if( nextStepsFlag ) {
                    Console.WriteLine( "\nMATING PAIRS AND OFFSPRING" );
                    int o = 0;
                    for( int i = 0; i < matingPairs.Count; i++ ) {
                        if( matingPairs[i].B == -1 ) {
                            //print single parent family
                            creatures[matingPairs[i].A].Print( matingPairs[i].A, "one", false, colors );
                            if( o < offspring.Count ) offspring[o++].Print( -1, "", true, colors );
                            if( o < offspring.Count ) offspring[o++].Print( -1, "", true, colors );
                        } else {
                            //print two parent family
                            creatures[matingPairs[i].A].Print( matingPairs[i].A, "mom", false, colors );
                            creatures[matingPairs[i].B].Print( matingPairs[i].B, "dad", false, colors );
                            if( o < offspring.Count ) offspring[o++].Print( -1, "", true, colors );
                            if( o < offspring.Count ) offspring[o++].Print( -1, "", true, colors );
                        }
                        Console.WriteLine();
                    }
                }
            }
        }

        // Creature class
        public class Creature
        {
            // static fields
            static int nextID = 100;

            // instance fields
            private int id;
            private int age;
            private double[] gene;
            private double fitness;
            private int[] geneSrc;      // 0=single parent, 1=mom, 2=dad
            private bool[] geneMut;     // true if mutated

            // public properties
            public int ID { get { return id; } }
            public int Age { get { return age; } set { age = value; } }
            public double[] Gene { get { return gene; } }
            public double[] GeneCopy { get { return Common.NewCopy( gene ); } }
            public double Fitness { get { return fitness; } set { fitness = value; } }
            public int[] GeneSrc { get { return geneSrc; } set { geneSrc = value; } }
            public bool[] GeneMut { get { return geneMut; } set { geneMut = value; } }

            // constructor, no gene history
            public Creature( double[] gene ) {
                this.id = nextID++;
                this.age = 0;
                this.gene = gene;
                this.fitness = -1;
                this.geneSrc = new int[gene.Length];
                this.geneMut = new bool[gene.Length];
            }

            // constructor, with gene history
            public Creature( double[] gene, int[] geneSrc, bool[] geneMut ) {
                this.id = nextID++;
                this.age = 0;
                this.gene = gene;
                this.fitness = -1;
                this.geneSrc = geneSrc;
                this.geneMut = geneMut;
            }

            // for sorting
            public static int FitnessDesending( Creature A, Creature B ) {
                return -1 * A.fitness.CompareTo( B.fitness );
            }

            // print creatrue
            public void Print( int index, string note, bool withColorFlag, Dictionary<string, ConsoleColor> colors ) {
                // assumes console colors are set to their default values

                // settings, variables
                const int genesPerRow = 100;
                int genesPrinted = 0;

                // calc abs max
                double max = Math.Max( gene.Max(), -gene.Min() );

                // print index
                if( index != -1 ) Console.Write( "{0:D2} ", index );
                else Console.Write( "   " );

                // print id
                Console.Write( "{0:D5} ", id );

                // print scale
                if( G.geneType == "double" )
                    Console.Write( "({0}) ", Common.FrmDbl( max, 4, "zeros-right", true ) );

                // print first gene chunk
                PrintGeneChunk( ref genesPrinted, genesPerRow, max, withColorFlag, colors );
                Console.Write( " " );

                // print age and fitness
                Console.Write( "a{0:D3} f{1} ", age, Common.FrmDbl( fitness, 8, "spaces-right" ) );

                // print note
                if( note == "mom" )
                    Console.ForegroundColor = colors["momBackground"];
                else if( note == "dad" )
                    Console.ForegroundColor = colors["dadBackground"];
                Console.WriteLine( note );
                Console.ForegroundColor = colors["defForeground"];

                // print remaing gene chuncks
                while( gene.Length > genesPrinted ) {
                    if( G.geneType == "double" ) Console.Write( "                " );
                    else Console.Write( "         " );
                    PrintGeneChunk( ref genesPrinted, genesPerRow, max, withColorFlag, colors );
                    Console.WriteLine( "" );
                }
            }

            // print next chunk of gene
            private void PrintGeneChunk( ref int genesPrinted, int genesPerRow, double max, bool withColorFlag, Dictionary<string, ConsoleColor> colors ) {

                // for each char in chunk
                int g; for( g = genesPrinted; g < Math.Min( gene.Length, genesPrinted + genesPerRow ); g++ ) {

                    // set colors
                    if( withColorFlag ) {
                        if( geneMut[g] )
                            Console.ForegroundColor = colors["mutForeground"];
                        else
                            Console.ForegroundColor = colors["brightForeground"];
                        switch( geneSrc[g] ) {
                            case 0: Console.BackgroundColor = colors["defBackground"]; break;
                            case 1: Console.BackgroundColor = colors["momBackground"]; break;
                            case 2: Console.BackgroundColor = colors["dadBackground"]; break;
                        }
                    }

                    // print character
                    switch( G.geneType ) {
                        case "binary":
                            Console.Write( (int)gene[g] );
                            break;
                        case "double":
                            Console.Write( Math.Min( 9, (int)( 5 + 5 * gene[g] / max ) ) );
                            break;
                    }
                }
                genesPrinted = g;

                // fix colors
                if( withColorFlag ) {
                    Console.ForegroundColor = colors["defForeground"];
                    Console.BackgroundColor = colors["defBackground"];
                }
            }

            // assign new values
            public void Assign( Creature other ) {
                this.id = other.id;
                this.age = other.age;
                this.gene = other.gene;             // does not copy data
                this.fitness = other.fitness;
                this.geneSrc = other.geneSrc;       // does not copy data
                this.geneMut = other.geneMut;       // does not copy data
            }
        }

        // Selection by fitness only class
        public class SelectionByFitness
        {
            static Int32 seed = 0;

            // method to select from list of creatures based on their fitness
            public static void Select( ref List<int> selected, List<SelectFromItem> selectFromList, int numToSelect,
                string method = "random", double bestWorstRatio = 5, bool preferBestFlag = true, bool showStepsFlag = false ) {

                // Data for creatures to select from must be loaded into selectFromList.
                // Assumes creatures are sorted from best to worst fitness.
                // Adds selected creature indexes to selected list
                // method = "best-worst"
                //   -chooses the best or worst from the list depending on preferBestFlag
                // method = "random"
                //   -chooses randomly from select from list
                //   -no other parameters used
                // method = "roulette"
                //   -chooses with ods based on fitness
                //   -preferBestFlag = true to prefer best fitness, false to prefer worst fitness
                // method = "rank"
                //   -chooses with odds based on order of creatures ranked best to worst
                //   -bestWorstRatio controls odds of best compared to odds of worst
                //   -preferBestFlag = true to prefer best fitness, false to prefer worst fitness
                // methoc = "map"
                //   -chooses by mapping fitness to a range from 1 to bestWorstRatio
                //   -bestWorstRatio controls odds of best compared to odds of worst
                //   -preferBestFlag = true to prefer best fitness, false to prefer worst fitness

                // working variables
                double highestSelectValue;
                double rankBase = 1, bestFitness, worstFitness;

                // check if wanting to select more than there are to select from
                if( numToSelect > selectFromList.Count ) throw new ArgumentException( "error: numToSelect > selectFromList.Count" );

                //fill selectValues with appropriate values depending on selection method
                switch( method ) {

                    case "best-worst":
                        // just takes the best or worst depending on preferBest
                        if( preferBestFlag )
                            for( int i = 0; i < numToSelect; i++ )
                                selected.Add( selectFromList[i].creatureIndex );
                        else
                            for( int i = selectFromList.Count - 1; i >= selectFromList.Count - numToSelect; i-- )
                                selected.Add( selectFromList[i].creatureIndex );
                        return;

                    case "random":
                        // even odds of any creature from the list
                        for( int i = 0; i < selectFromList.Count; i++ ) {
                            selectFromList[i].selectSize = 1;
                            selectFromList[i].selectValue = i + 1;
                        }
                        highestSelectValue = selectFromList.Count;
                        break;

                    case "roulette":
                        // roulette based on creatureFitness (odds proportional to fitness or 1/fitness)
                        highestSelectValue = 0;
                        for( int i = 0; i < selectFromList.Count; i++ ) {
                            selectFromList[i].selectSize = preferBestFlag ? selectFromList[i].creatureFitness : 1 / selectFromList[i].creatureFitness;
                            selectFromList[i].selectValue = highestSelectValue += selectFromList[i].selectSize;
                        }
                        break;

                    case "rank":
                        // rank selection based on order of creatures and ratio between best odds and worst odds
                        // b^(n-1)=ratio  =>  b=ratio^(1/(n-1))
                        rankBase = Math.Pow( bestWorstRatio, 1 / (double)( selectFromList.Count - 1 ) );
                        highestSelectValue = selectFromList[0].selectValue = selectFromList[0].selectSize = preferBestFlag ? bestWorstRatio : 1;
                        for( int i = 1; i < selectFromList.Count; i++ ) {
                            selectFromList[i].selectSize = preferBestFlag ? selectFromList[i - 1].selectSize / rankBase : selectFromList[i - 1].selectSize * rankBase;
                            selectFromList[i].selectValue = highestSelectValue += selectFromList[i].selectSize;
                        }
                        break;

                    case "map":
                        // selection by mapping fitness to range from 1 to bestWorstRatio
                        // b^(best fitness - worst fitness) = ratio
                        // => b = ratio^(1/(best-worst)
                        // size = b^(fitness-worst)  or size = b^(best-fitness)
                        bestFitness = selectFromList[0].creatureFitness;
                        worstFitness = selectFromList.Last().creatureFitness;
                        rankBase = Math.Pow( bestWorstRatio, 1 / ( bestFitness - worstFitness ) );
                        highestSelectValue = 0;
                        for( int i = 0; i < selectFromList.Count; i++ ) {
                            selectFromList[i].selectSize = Math.Pow( rankBase, preferBestFlag ? ( selectFromList[i].creatureFitness - worstFitness ) : ( bestFitness - selectFromList[i].creatureFitness ) );
                            selectFromList[i].selectValue = highestSelectValue += selectFromList[i].selectSize;
                        }
                        break;

                    default:
                        throw new ArgumentException( "bad method string, " + method );
                }

                // debug print
                if( showStepsFlag ) ShowSelection( method, rankBase, selectFromList, highestSelectValue, selected );

                // get random select value, select from list, remove creature from list, repeat
                Random random = new Random( seed++ );
                double r; int s; double toSubtract;
                for( int i = 0; i < numToSelect; i++ ) {

                    // calc random number from 0 to highestSelectValue
                    r = random.NextDouble() * highestSelectValue;

                    // find first selectValue equal to or greater than random, add this one to the output list
                    s = 0; while( selectFromList[s].selectValue < r ) s++;
                    selected.Add( selectFromList[s].creatureIndex );

                    // remove this creature from the select from list and decrease subsequent selectValues
                    toSubtract = selectFromList[s].selectSize;
                    selectFromList.RemoveAt( s );
                    for( int j = s; j < selectFromList.Count; j++ ) selectFromList[j].selectValue -= toSubtract;
                    highestSelectValue -= toSubtract;

                    // show progress
                    if( showStepsFlag ) ShowSelection( method, rankBase, selectFromList, highestSelectValue, selected );

                }
            }

            public class SelectFromItem
            {
                public readonly int creatureIndex;
                public readonly double creatureFitness;
                public double selectSize;
                public double selectValue;

                public SelectFromItem( int creatureIndex, double creatureFitness ) {
                    this.creatureIndex = creatureIndex;
                    this.creatureFitness = creatureFitness;
                    this.selectSize = 0;
                    this.selectValue = 0;
                }
            }

            static void ShowSelection( string method, double rankBase, List<SelectFromItem> selectFromList, double highestSelectValue, List<int> toMate ) {

                Console.WriteLine( "\nSelect by {0}", method.ToUpper() );

                // method specfic printing
                if( method == "rank" || method == "map" ) Console.WriteLine( "rankBase={0}", rankBase );

                // for each selectFrom item printing
                for( int i = 0; i < selectFromList.Count; i++ )
                    Console.WriteLine( "index={0:D2}  fitness={1}  selectSize={2}  selectValue={3}",
                        selectFromList[i].creatureIndex,
                        Common.FrmDbl( selectFromList[i].creatureFitness, 6, "zeros-right" ),
                        Common.FrmDbl( selectFromList[i].selectSize, 6, "zeros-right" ),
                        Common.FrmDbl( selectFromList[i].selectValue, 6, "zeros-right" ) );

                // printing at the end
                Console.WriteLine( " use random select values from 0 to {0}", highestSelectValue );

                // selected so far
                Console.Write( "Selected so far: " );
                for( int i = 0; i < toMate.Count; i++ ) Console.Write( "{0}, ", toMate[i] );
                Console.WriteLine(); Console.Write( "continue?" ); Console.ReadKey( true ); Console.WriteLine();

            }
        }

        public struct IntPair
        {
            public int A;
            public int B;
            public IntPair( int A, int B ) {
                this.A = A;
                this.B = B;
            }
        }

        // Delegates for (1) gene length, (2)make random gene, and (3) fitness function
        public delegate double[] RandomGeneSig();
        public delegate bool FitnessSig( double[][] genes, ref double[] fitnessAry, out double bestFitness, out double avrFitness );
    }

    // Feed Forward Neural Network
    class FFNN
    {
        // Functions that must be fast
        // - evalulate outputs from inputs
        // - create new nets from a weight vector
        // Other required functions
        // - calc number of weights needed for defintion
        // - return vector of random weights
        // - return weights
        // - return definition
        // - randomize
        // Optional functions
        // - set weights
        // - print network

        // static fields
        private static Int32 seed = 0;
        private static Random random = new Random( seed++ );
        private static readonly double randomScale = 2.0;
        private static readonly double randomOffset = -1.0;

        // instance fields
        private int[] def;              // numInputs, numNeurons, numNeurons,,,
        private double[] weights;       // bias, weight, weight,,, bias, weight, weight,,,

        // instance properties
        public int[] Def { get { return NewCopy( def ); } }
        public double[] Weights { get{ return weights; } set{ weights = value; }}
        public double[] WeightsCopy { get { return NewCopy( weights ); } set { weights = NewCopy( value ); } }

        // calculates number of required weights for a FFNN defintion
        public static int NumberOfWeights( int[] def ) {
            int numOut = 0;
            for( int i = 1; i < def.Length; i++ )
                numOut += def[i] * ( def[i - 1] + 1 );
            return numOut;
        }

        // return a random vector of weights
        public static double[] RandomWeights( int[] def ) {
            double[] weightsOut = new double[NumberOfWeights( def )];
            for( int i = 0; i < weightsOut.Length; i++ )
                weightsOut[i] = random.NextDouble() * randomScale + randomOffset;
            return weightsOut;
        }

        // construct a neural net given a def with blank weights
        public FFNN( int[] def ) {
            this.def = NewCopy( def );
            this.weights = new double[ NumberOfWeights( def ) ];
        }

        // construct a neural net given a def and weight vector
        public FFNN( int[] def, double[] weights ) {
            // assumes the length of weights is correct
            this.def = NewCopy( def );
            this.weights = NewCopy( weights );
        }

        // calculate output vector from input vector
        public double[] Output( double[] INs ) {
            // assumes length of inputs is correct

            //pointers into weights[]
            int w = 0;

            // for each layer i starting with i=1
            for( int i = 1; i < def.Length; i++ ) {

                // create a new layer output vector
                double[] OUTs = new double[def[i]];

                // for each neuron in this layer
                for( int j = 0; j < def[i]; j++ ) {

                    // start with the bias (or threshold)
                    OUTs[j] = ( -1 ) * weights[w++];

                    // add each input times weight
                    for( int k = 0; k < def[i - 1]; k++ )
                        OUTs[j] += INs[k] * weights[w++];

                    // put result through sigmoid function
                    OUTs[j] = Smoothing( OUTs[j], 1.0 );
                }
                // use these outputs as the next inputs
                INs = OUTs;
            }
            return INs;
        }

        // smoothing function
        double Smoothing( double netinput, double response ) {
            return ( 1 / ( 1 + Math.Exp( -netinput / response ) ) );
        }

        // re-randomize
        public static void Randomize() {
            random = new Random( seed++ );
        }

        // print network
        public override string ToString() {

            StringBuilder outStr = new StringBuilder( String.Format( "\n LAYER=0, {0} inputs\n", def[0] ) );
            for( int i = 0; i < def[0]; i++ )
                outStr.Append( String.Format( "  n(0,{0})\n", i + 1 ) );

            int w = 0;
            for( int i = 1; i < def.Length; i++ ) {

                outStr.Append( String.Format( "\n LAYER={0}, {1} neurons\n", i, def[i] ) );

                // for each neuron in this layer
                for( int j = 0; j < def[i]; j++ ) {

                    outStr.Append( String.Format( "  n({0},{1}) = Sigmoid( ({2:F2}) * (-1)", i, j + 1, weights[w++] ) );

                    // for each input
                    for( int k = 0; k < def[i - 1]; k++ )
                        outStr.Append( String.Format( " + ({0:F2}) * n({1},{2})", weights[w++], i - 1, k + 1 ) );

                    outStr.Append( " )\n" );
                }
            }
            return outStr.ToString();
        }

        // int[] copy function
        public static int[] NewCopy( int[] toCopy ) {
            int[] copy = new int[toCopy.Length];
            Array.Copy( toCopy, copy, toCopy.Length );
            return copy;
        }

        // double[] copy function
        public static double[] NewCopy( double[] toCopy ) {
            double[] copy = new double[toCopy.Length];
            Array.Copy( toCopy, copy, toCopy.Length );
            return copy;
        }
    }

    // TicTacToe Class
    public class TicTacToe
    {
        // 

        public static double[] RoundRobin( double[,] dary ) {

            return new double[] { 1.0, 2.0 };
        }

    }

    // Common functions
    public class Common
    {
        // swap int
        public static void Swap( ref int A, ref int B ) {
            int C = A;
            A = B; B = C;
        }

        // copy int[]
        public static int[] NewCopy( int[] toCopy ) {
            int[] copy = new int[toCopy.Length];
            Array.Copy( toCopy, copy, toCopy.Length );
            return copy;
        }

        // copy double[]
        public static double[] NewCopy( double[] toCopy ) {
            double[] copy = new double[toCopy.Length];
            Array.Copy( toCopy, copy, toCopy.Length );
            return copy;
        }

        // copy bool[]
        static bool[] NewCopy( bool[] toCopy ) {
            bool[] copy = new bool[toCopy.Length];
            Array.Copy( toCopy, copy, toCopy.Length );
            return copy;
        }

        // format double using int len number of characters
        public static string FrmDbl( double n, int len, string padding = "zeros-right", bool truncateFlag = false ) {

            // padding options are "spaces-left", "spaces-right", "zeros-left", "zeros-right"

            // get string and find decimal point
            string outStr = n.ToString();
            int i = outStr.IndexOf( '.' );
            if( i == -1 ) { i = outStr.Length; }

            // if number is too big
            if( i > len )
                if( truncateFlag )
                    return outStr = "".PadLeft( len, '+' );
                else
                    return outStr.Substring( 0, i );

            // else if decimal part of number makes it too long
            if( outStr.Length >= len )
                return outStr.Substring( 0, len );

            // do padding
            switch( padding ) {

                case "spaces-left":
                case "left-spaces":
                    return outStr.PadLeft( len );

                case "spaces-right":
                case "right-spaces":
                    return outStr.PadRight( len );

                case "zeros-left":
                case "left-zeros":
                    if( outStr[0] == '-' )
                        return '-' + outStr.Substring( 1 ).PadLeft( len - 1, '0' );
                    else
                        return outStr.PadLeft( len, '0' );

                case "zeros-right":
                case "right-zeros":
                    if( i < outStr.Length )
                        return outStr.PadRight( len, '0' );
                    else
                        return ( outStr + '.' ).PadRight( len, '0' );
            }

            throw new ArgumentException( "bad padding = " + padding );
        }

    }
}
