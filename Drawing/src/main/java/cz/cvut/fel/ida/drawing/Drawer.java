package cz.cvut.fel.ida.drawing;


import cz.cvut.fel.ida.setup.DrawingStyle;
import cz.cvut.fel.ida.setup.Settings;

import javax.swing.*;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.NumberFormat;
import java.util.Locale;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Created by gusta on 8.3.17.
 */
public abstract class Drawer<S> {   //todo next replace hashcodes (which collide sometimes) with truly unique IDs

    private static final Logger LOG = Logger.getLogger(Drawer.class.getName());

    public NumberFormat numberFormat;
    protected DrawingStyle style;
    protected DotBuilder dot;
    protected Settings.Detail drawingDetail;
    protected boolean storeNotShow;

    protected GraphViz graphviz;
    public Settings settings;

    public Drawer(Settings settings) {
        this.settings = settings;
        this.style = settings.drawingStyle != null ? settings.drawingStyle : new DrawingStyle();
        this.dot = new DotBuilder(style);

        if (settings.drawing) {
            this.graphviz = new GraphViz(settings);

            Logger.getLogger("java.awt").setLevel(Level.WARNING);
            Logger.getLogger("sun.awt").setLevel(Level.WARNING);
            Logger.getLogger("javax.swing").setLevel(Level.WARNING);
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (ClassNotFoundException | InstantiationException | IllegalAccessException | UnsupportedLookAndFeelException ex) {
            }
        }

        this.drawingDetail = settings.drawingDetail;
        this.numberFormat = createNumberFormat();
        this.storeNotShow = settings.storeNotShow;
    }

    /**
     * Builds the {@link NumberFormat} used for drawing values from the configurable
     * {@link DrawingStyle#valueDecimals} and {@link DrawingStyle#displayValues} flags.
     * Returns null when values should be hidden entirely.
     */
    protected NumberFormat createNumberFormat() {
        if (drawingDetail == Settings.Detail.LOW || !style.displayValues) {
            return null;
        }
        DecimalFormatSymbols symbols = new DecimalFormatSymbols(Locale.US);
        symbols.setInfinity("Infinity");
        String pattern = style.valueDecimals <= 0 ? "#" : "#." + repeat('#', style.valueDecimals);
        return new DecimalFormat(pattern, symbols);
    }

    private String repeat(char c, int count) {
        StringBuilder sb = new StringBuilder(count);
        for (int i = 0; i < count; i++) {
            sb.append(c);
        }
        return sb.toString();
    }

    /**
     * Formats a numeric value with the current style precision.
     */
    public String formatNumber(double value) {
        if (numberFormat == null) {
            return "";
        }
        return numberFormat.format(value);
    }

    /**
     * Opens a graph and emits the graph-level attributes from {@link DrawingStyle#graph}.
     */
    protected void startStyledGraph() {
        graphviz.start_graph();
        addGraphDefaults();
    }

    /**
     * Emits graph-level attributes from {@link DrawingStyle#graph} (call right after the graph is opened).
     */
    protected void addGraphDefaults() {
        for (Map.Entry<String, String> attribute : style.graph.entrySet()) {
            graphviz.addln(attribute.getKey() + "=" + DotBuilder.quote(attribute.getValue()) + ";");
        }
    }

    public void display(byte[] imageBytes, String name) {
        //now turn off annoying java awt logging messages here
        Logger.getLogger("java.awt").setLevel(Level.WARNING);
        Logger.getLogger("sun.awt").setLevel(Level.WARNING);
        Logger.getLogger("javax.swing").setLevel(Level.WARNING);

        Object lock = new Object();
        DrawWindow debugWindow = new DrawWindow(imageBytes, name, lock);
        debugWindow.pauseUntilWindowCloses(lock);
    }

    public void draw(S obj) {
        if (graphviz == null) {
            LOG.warning("Drawing is required but no graphviz exec found, could not draw!");
            return;
        }
        this.graphviz.clearGraph();
        loadGraph(obj);
        try {
            LOG.info("Paused for drawing with Graphviz...(this may take a while for bigger graphs)");
            if (storeNotShow) {
                graphviz.storeGraphSource(obj.toString());
                graphviz.storeGraphImage(obj.toString());
                LOG.info("Graph stored into file named: " + obj.toString());
            } else if (graphviz.isDisplayableType()) {
                display(graphviz.getGraphImage(obj.toString()), obj.toString());
            } else {
                LOG.info("Image type '" + graphviz.imgtype + "' cannot be displayed; storing the graph instead");
                graphviz.storeGraphSource(obj.toString());
                graphviz.storeGraphImage(obj.toString());
                LOG.info("Graph stored into file named: " + obj.toString());
            }
        } catch (IOException | InterruptedException e) {
            LOG.severe(e.getMessage());
        }
    }

    public byte[] drawIntoFile(S obj, String path) {
        if (this.graphviz == null) {
            this.graphviz = new GraphViz(this.settings);
        }

        this.graphviz.clearGraph();
        loadGraph(obj);

        try {
            graphviz.renderToFile(new File(path));
        } catch (IOException | InterruptedException e) {
            LOG.severe(e.getMessage());
        }

        return null;
    }

    public byte[] drawIntoBytes(S obj) {
        if (this.graphviz == null) {
            this.graphviz = new GraphViz(this.settings);
        }

        this.graphviz.clearGraph();
        loadGraph(obj);

        return graphviz.getGraphUsingTemporaryFile(graphviz.getDotSource(), graphviz.imgtype, graphviz.algorithm);
    }

    public String getGraphSource(S obj) {
        if (this.graphviz == null) {
            this.graphviz = new GraphViz(this.settings);
        }

        this.graphviz.clearGraph();
        loadGraph(obj);

        return this.graphviz.getDotSource();
    }

    /**
     * Returns the graph of the given object as GraphML, an XML format that can be
     * loaded directly into networkx, e.g.:
     * <pre>import networkx as nx; nx.read_graphml(...)</pre>
     *
     * @return GraphML string, or null if the graphviz-based conversion failed.
     */
    public String getGraphMLSource(S obj) {
        if (this.graphviz == null) {
            this.graphviz = new GraphViz(this.settings);
        }

        this.graphviz.clearGraph();
        loadGraph(obj);

        return this.graphviz.toGraphML();
    }

    /**
     * Stores the graph of the given object as a GraphML file consumable by networkx.
     *
     * @param obj  the object whose graph should be exported
     * @param path destination file path (parent directories are created if needed)
     */
    public void saveGraphML(S obj, String path) {
        String graphML = getGraphMLSource(obj);
        if (graphML == null) {
            LOG.warning("Could not create GraphML source for " + obj);
            return;
        }

        File file = new File(path);
        if (file.getParentFile() != null) {
            file.getParentFile().mkdirs();
        }
        try (FileWriter writer = new FileWriter(file)) {
            writer.write(graphML);
        } catch (IOException e) {
            LOG.severe(e.getMessage());
        }
    }

    public abstract void loadGraph(S obj);  //todo add indentation into the dot file

}
