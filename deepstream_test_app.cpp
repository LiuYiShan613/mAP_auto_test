/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime_api.h>

#include "diva_yml_parser.h"
#include "nvds_opticalflow_meta.h"
#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include "gst-nvmessage.h"

#include <algorithm>
#include <gst/gst.h>
#include "nvbufsurface.h"
//#include "nvds_meta.h"
#include "nvdsinfer.h"
#include "gstnvdsmeta.h"

#include <opencv2/opencv.hpp> 
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <regex>

using namespace cv; 


/* NVIDIA Decoder source pad memory feature. This feature signifies that source
 * pads having this capability will push GstBuffers containing cuda buffers. */
#define GST_CAPS_FEATURES_NVMM "memory:NVMM"

/* Inference engine config file path*/
#define PGIE_CONFIG_FILE    "diva_test_pgie_config.yml"

/* Set the user metadata type */
// #define DIVA_STREAM_META (nvds_get_user_meta_type(((gchar *)"DIVA.STREAM.META")))

gchar * config_path = NULL;
float bbox_all;

// To extract video name
std::string extractVideoName(const std::string& ymlFilePath) {
    std::ifstream ymlFile(ymlFilePath);
    if (!ymlFile.is_open()) {
        throw std::runtime_error("Failed to open YML file: " + ymlFilePath);
    }

    std::string line;
    std::regex listRegex(R"(list:\s*file://.*/(.*)\.mp4)");
    std::smatch match;

    while (std::getline(ymlFile, line)) {
        if (std::regex_search(line, match, listRegex)) {
            ymlFile.close();
            return match[1]; // Return the extracted video name
        }
    }

    ymlFile.close();
    throw std::runtime_error("Failed to find 'list' in YML file.");
}

static GstPadProbeReturn
pad_probe (GstPad * pad, GstPadProbeInfo *info, gpointer u_data)
{
  float bbox_avg = 0;
  int t_frame;
  
  std::string ymlFilePath = "diva_test_config.yml";

  std::string videoName;

  try {
      videoName = extractVideoName(ymlFilePath);
  } catch (const std::exception& e) {
      g_print("Error: %s\n", e.what());
      return GST_PAD_PROBE_OK;
  }

  std::string outputFileName = "auto_test/" + videoName + ".txt";

  // to write in file
  FILE *file = fopen(outputFileName.c_str(), "a");
  // FILE *file = fopen("output.txt", "a");
  if (file == NULL) {
    g_print("Error: Unable to open output.txt for writing\n");
    return GST_PAD_PROBE_OK;
  }

  GstBuffer *buf = (GstBuffer *) info->data;
  NvDsMetaList *l_frame = NULL;

  NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta (buf);
  
  std::vector<std::tuple<int, std::vector<std::vector<int>>>> frame_data_list;;

  for (l_frame = batch_meta->frame_meta_list; l_frame != NULL;
      l_frame = l_frame->next){
    gint i = 0;
    NvDsObjectMetaList *l_obj = NULL;
    NvDsFrameMeta *frame_meta = (NvDsFrameMeta *) (l_frame->data);
    
    std::vector<std::vector<int>> bbox_coords;
    // every frame
    for (l_obj = frame_meta->obj_meta_list; l_obj != NULL; l_obj = l_obj->next){
      NvDsObjectMeta *obj_meta = (NvDsObjectMeta *) (l_obj->data);
      NvOSD_RectParams & rect_params = obj_meta->rect_params;
      
      // to call in python
      bbox_coords.push_back({
          (int)rect_params.left,
          (int)rect_params.top,
          (int)rect_params.width + (int)rect_params.left,  // xmax
          (int)rect_params.height + (int)rect_params.top   // ymax
      });

      i++;
    }

    frame_data_list.push_back(std::make_tuple(frame_meta->frame_num, bbox_coords));

    // g_print("Stream num: %d  Frame num: %d  NTP: %ld  Bbox num: %d\n",
    //     frame_meta->source_id, frame_meta->frame_num, frame_meta->ntp_timestamp, i);
    
    //write file
    fprintf(file, "Stream num: %d  Frame num: %d  NTP: %ld  Bbox num: %d\n",
        frame_meta->source_id, frame_meta->frame_num, frame_meta->ntp_timestamp, i);
    
    t_frame = frame_meta->frame_num;
    
    bbox_all += i;

  }

  std::sort(frame_data_list.begin(), frame_data_list.end(),
              [](const auto &a, const auto &b) {
                  return std::get<0>(a) < std::get<0>(b); // sort by frame_num
              });

  for (const auto &frame_data : frame_data_list) {
        int frame_num = std::get<0>(frame_data);
        const auto &bbox_coords = std::get<1>(frame_data);
        // fprintf(file, "Frame num: %d\n",frame_num);
        for (const auto &bbox : bbox_coords) {
            fprintf(file, "%d,%d,%d,%d\n", bbox[0], bbox[1], bbox[2], bbox[3]);
        }
        //write file
        // fprintf(file, "Stream num: %d  Frame num: %d  NTP: %ld  Bbox num: %d\n",
        // frame_meta->source_id, frame_meta->frame_num, frame_meta->ntp_timestamp, i);
    }
  fclose(file);

  return GST_PAD_PROBE_OK;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_WARNING:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_warning (msg, &error, &debug);
      g_printerr ("WARNING from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      g_free (debug);
      g_printerr ("Warning: %s\n", error->message);
      g_error_free (error);
      break;
    }
    case GST_MESSAGE_ERROR:
    {
      gchar *debug;
      GError *error;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n",
          GST_OBJECT_NAME (msg->src), error->message);
      if (debug)
        g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    case GST_MESSAGE_ELEMENT:
    {
      if (gst_nvmessage_is_stream_eos (msg)) {
        guint index;
        if (gst_nvmessage_parse_stream_eos (msg, &index)) {
        }
      }
      break;
    }
    default:
      break;
  }
  return TRUE;
}

static void
cb_newpad (GstElement * decodebin, GstPad * decoder_src_pad, gpointer data)
{
  GstCaps *caps = gst_pad_get_current_caps (decoder_src_pad);
  if (!caps) {
    caps = gst_pad_query_caps (decoder_src_pad, NULL);
  }
  const GstStructure *str = gst_caps_get_structure (caps, 0);
  const gchar *name = gst_structure_get_name (str);
  GstElement *source_bin = (GstElement *) data;
  GstCapsFeatures *features = gst_caps_get_features (caps, 0);

  /* Need to check if the pad created by the decodebin is for video and not
   * audio. */
  if (!strncmp (name, "video", 5)) {
    /* Link the decodebin pad only if decodebin has picked nvidia
     * decoder plugin nvdec_*. We do this by checking if the pad caps contain
     * NVMM memory features. */
    if (gst_caps_features_contains (features, GST_CAPS_FEATURES_NVMM)) {
      /* Get the source bin ghost pad */
      GstPad *bin_ghost_pad = gst_element_get_static_pad (source_bin, "src");
      if (!gst_ghost_pad_set_target (GST_GHOST_PAD (bin_ghost_pad),
              decoder_src_pad)) {
        g_printerr ("Failed to link decoder src pad to source bin ghost pad\n");
      }
      gst_object_unref (bin_ghost_pad);
    } else {
      g_printerr ("Error: Decodebin did not pick nvidia decoder plugin.\n");
    }
  }
}

static void
decodebin_child_added (GstChildProxy * child_proxy, GObject * object,
    gchar * name, gpointer user_data)
{
  if (g_strrstr (name, "decodebin") == name) {
    g_signal_connect (G_OBJECT (object), "child-added",
        G_CALLBACK (decodebin_child_added), user_data);
  } else if (g_strrstr (name, "nvv4l2decoder") == name) {
    parse_decoder_yaml (object, config_path, "nvv4l2decoder");
  }
}

static GstElement *
create_source_bin (guint index, gchar * uri)
{
  GstElement *bin = NULL, *uri_decode_bin = NULL;
  gchar bin_name[32] = { };

  g_snprintf (bin_name, 31, "source-bin-%02d", index);
  /* Create a source GstBin to abstract this bin's content from the rest of the
   * pipeline */
  bin = gst_bin_new (bin_name);

  /* Source element for reading from the uri.
   * We will use decodebin and let it figure out the container format of the
   * stream and the codec and plug the appropriate demux and decode plugins. */
  uri_decode_bin = gst_element_factory_make ("uridecodebin", "uri-decode-bin");

  if (!bin || !uri_decode_bin) {
    g_printerr ("One element in source bin could not be created.\n");
    return NULL;
  }

  /* We set the input uri to the source element */
  g_object_set (G_OBJECT (uri_decode_bin), "uri", uri, NULL);

  /* Connect to the "pad-added" signal of the decodebin which generates a
   * callback once a new pad for raw data has beed created by the decodebin */
  g_signal_connect (G_OBJECT (uri_decode_bin), "pad-added",
      G_CALLBACK (cb_newpad), bin);
  g_signal_connect (G_OBJECT (uri_decode_bin), "child-added",
      G_CALLBACK (decodebin_child_added), bin);

  gst_bin_add (GST_BIN (bin), uri_decode_bin);

  /* We need to create a ghost pad for the source bin which will act as a proxy
   * for the video decoder src pad. The ghost pad will not have a target right
   * now. Once the decode bin creates the video decoder and generates the
   * cb_newpad callback, we will set the ghost pad target to the video decoder
   * src pad. */
  if (!gst_element_add_pad (bin, gst_ghost_pad_new_no_target ("src",
              GST_PAD_SRC))) {
    g_printerr ("Failed to add ghost pad in source bin\n");
    return NULL;
  }

  return bin;
}

int
main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *streammux = NULL, 
      *pgie = NULL, *nvof = NULL, *nvdslogger = NULL, *tiler = NULL,
      *nvvidconv = NULL, *transform = NULL, *nvosd = NULL, *sink = NULL,
      *queue1, *queue2, *queue3, *queue4, *queue5, *queue6;
  GstElement *divatracking = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  guint i = 0, num_sources = 0;
  guint pgie_batch_size;
  GList *temp, *src_list = NULL;
  gboolean DISPLAY = display_parse (argv[1], "variable");

  config_path = argv[1];

  int current_device = -1;
  cudaGetDevice(&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, current_device);

  /* Check input arguments */
  if (argc < 2) {
    g_printerr ("Usage: %s <yml file>\n", argv[0]);
    return -1;
  }

  /* Standard GStreamer initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* Create gstreamer elements */
  /* Create Pipeline element that will form a connection of other elements */
  pipeline = gst_pipeline_new ("diva-pipeline");

  /* Create nvstreammux instance to form batches from one or more sources. */
  streammux = gst_element_factory_make ("nvstreammux", "streammux");

  if (!pipeline || !streammux) {
    g_printerr ("Pipeline and streammux could not be created. Exiting.\n");
    return -1;
  }
  gst_bin_add_many (GST_BIN (pipeline), streammux, NULL);

  nvds_parse_source_list(&src_list, argv[1], "source-list");
  temp = src_list;
  while(temp) {
    num_sources++;
    temp=temp->next;
  }
  g_list_free(temp);

  for (i = 0; i < num_sources; i++) {
    GstPad *sinkpad, *srcpad;
    gchar pad_name[16] = { };

    GstElement *source_bin= NULL;
    if (g_str_has_suffix (argv[1], ".yml") || g_str_has_suffix (argv[1], ".yaml")) {
      g_print("Now playing : %s\n",(char*)(src_list)->data);
      source_bin = create_source_bin (i, (char*)(src_list)->data);
    } else {
      source_bin = create_source_bin (i, argv[i + 1]);
    }
    if (!source_bin) {
      g_printerr ("Failed to create source bin. Exiting.\n");
      return -1;
    }

    gst_bin_add (GST_BIN (pipeline), source_bin);

    g_snprintf (pad_name, 15, "sink_%u", i);
    sinkpad = gst_element_get_request_pad (streammux, pad_name);
    if (!sinkpad) {
      g_printerr ("Streammux request sink pad failed. Exiting.\n");
      return -1;
    }

    srcpad = gst_element_get_static_pad (source_bin, "src");
    if (!srcpad) {
      g_printerr ("Failed to get src pad of source bin. Exiting.\n");
      return -1;
    }

    if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link source bin to stream muxer. Exiting.\n");
      return -1;
    }

    gst_object_unref (srcpad);
    gst_object_unref (sinkpad);

    if (g_str_has_suffix (argv[1], ".yml") || g_str_has_suffix (argv[1], ".yaml")) {
      src_list = src_list->next;
    }
  }

  g_list_free(src_list);

  /* Use nvinfer to infer on batched frame. */
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");

  /* NVOF */
  nvof = gst_element_factory_make ("nvof", "Optical-Flow");
  nvds_parse_nvof (nvof, argv[1], "nvof");

  /* Add queue elements between elements */
  queue1 = gst_element_factory_make ("queue", "queue1");

  queue2 = gst_element_factory_make ("queue", "queue2");

  queue3 = gst_element_factory_make ("queue", "queue3");

  queue4 = gst_element_factory_make ("queue", "queue4");

  queue5 = gst_element_factory_make ("queue", "queue5");

  queue6 = gst_element_factory_make ("queue", "queue6");

  /* Add DIVATracking element */
  divatracking = gst_element_factory_make ("divatracking", "divatracking");
  diva_parse_tracking (divatracking, argv[1], "divatracking");
  
  nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger");

  if (DISPLAY) {
    tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");
    nvds_parse_tiler(tiler, argv[1], "tiler");
    nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
    nvds_parse_nvvidconv(nvvidconv, argv[1], "nvvidconv");
    nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
    nvds_parse_osd(nvosd, argv[1], "osd");
    sink = gst_element_factory_make ("nveglglessink", "nvvideo-renderer");
    nvds_parse_egl_sink (sink, argv[1], "sink");
    if (prop.integrated){
      transform = gst_element_factory_make ("nvegltransform", "egl-transform");
      if (!transform){
        g_printerr ("Transform element could not be created, Exiting.\n");
        return -1;
      }
    }
  } else {
    sink = gst_element_factory_make ("fakesink", "fakesink");
  }

  if (!pgie || !divatracking || !nvdslogger || !sink ||
      !queue1 || !queue2 || !queue3 || !queue4 || !queue5 || !queue6) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  nvds_parse_streammux(streammux, argv[1], "streammux");

  g_object_set (G_OBJECT (pgie), "config-file-path", PGIE_CONFIG_FILE, NULL);
  g_object_get (G_OBJECT (pgie), "batch-size", &pgie_batch_size, NULL);
  if (pgie_batch_size != num_sources) {
    g_printerr ("WARNING: Overriding infer-config batch-size (%d) with number of sources (%d)\n",
        pgie_batch_size, num_sources);
    g_object_set (G_OBJECT (pgie), "batch-size", num_sources, NULL);
  }
  
  /* we add a message handler */
  bus = gst_pipeline_get_bus (GST_PIPELINE (pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  /* Set up the pipeline */
  /* we add all elements into the pipeline */
  /* Jetson user and want to display, please add plugin into this section */
  if (transform && DISPLAY){
    /* Display */
    gst_bin_add_many (GST_BIN (pipeline), pgie, nvof, divatracking, nvdslogger,
        tiler, nvvidconv, nvosd, transform, sink,
        queue1, queue2, queue3, queue4, queue5, queue6, NULL);
    /* we link the elements together */
    /* nvstreammux -> pgie -> nvof -> divatracking -> nvdslogger -> nvtiler ->
        nvvidconv -> nvosd -> transform -> video-renderer */
    /* If user want to display, add plugin into this pipeline */
    if (!gst_element_link_many (streammux, queue1, pgie, queue2, nvof, queue3,
        divatracking, queue4, nvdslogger, tiler, queue5, nvvidconv, nvosd,
        queue6, transform, sink, NULL)){
      g_printerr ("Elements in pipeline could not be linked. Exiting.\n");
      return -1;
    }
  /* PC user and want to display, please add plugin into this section */
  } else if (!transform && DISPLAY) {
    /* Display */
    gst_bin_add_many (GST_BIN (pipeline), pgie, nvof, divatracking, nvdslogger,
        tiler, nvvidconv, nvosd, sink,
        queue1, queue2, queue3, queue4, queue5, queue6, NULL);
    /* we link the elements together */
    /* nvstreammux -> pgie -> nvof -> divatracking -> nvdslogger -> nvtiler ->
        nvvidconv -> nvosd -> video-renderer */
    /* If user want to display, add plugin into this pipeline */
    if (!gst_element_link_many (streammux, queue1, pgie, queue2, nvof, queue3, divatracking,
        queue4, nvdslogger, tiler, queue5, nvvidconv, nvosd, queue6, sink, NULL)){
      g_printerr ("Elements in pipeline could not be linked. Exiting.\n");
      return -1;
    }
  /* PC and Jetson user but don't want to display, please add plugin into this section */
  } else {
    /* Undisplay */
    gst_bin_add_many (GST_BIN (pipeline), pgie, /*nvof, divatracking, */nvdslogger,
        sink, queue1, queue2, queue3, queue4, NULL);
    /* we link the elements together */
    /* nvstreammux -> pgie -> nvof -> divatracking -> nvdslogger */
    /* If user want to display, add plugin into this pipeline */
    if (!gst_element_link_many (streammux, queue1, pgie, queue2, /*nvof, queue3, divatracking,*/
        queue4, nvdslogger, sink, NULL)){
      g_printerr ("Elements in pipeline could not be linked. Exiting.\n");
      return -1;
    }
  }
  
  /* Lets add probe to get informed of the meta data generated, we add probe to
   * the sink pad of the osd element, since by that time, the buffer would have
   * had got all the metadata. */
  GstPad *diva_src_pad = NULL;

  diva_src_pad = gst_element_get_static_pad (nvdslogger, "src");
  if (!diva_src_pad)
    g_print ("Unable to get diva src pad\n");
  else
    gst_pad_add_probe (diva_src_pad, GST_PAD_PROBE_TYPE_BUFFER,
        pad_probe, NULL, NULL);
  gst_object_unref (diva_src_pad);
  
  /* Set the pipeline to "playing" state */
  if (g_str_has_suffix (argv[1], ".yml") || g_str_has_suffix (argv[1], ".yaml")) {
    g_print ("Using file: %s\n", argv[1]);
  }
  else {
    g_print ("Now playing:");
    for (i = 0; i < num_sources; i++) {
      g_print (" %s,", argv[i + 1]);
    }
    g_print ("\n");
  }
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait till pipeline encounters an error or EOS */
  g_print ("Running...\n");
  g_main_loop_run (loop);

  /* Out of the main loop, clean up nicely */
  g_print ("Returned, stopping playback\n");
  gst_element_set_state (pipeline, GST_STATE_NULL);
  g_print ("Deleting pipeline\n");
  gst_object_unref (GST_OBJECT (pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  return 0;
}
