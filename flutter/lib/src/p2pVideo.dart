import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
// import 'package:flutterwebrtc/src/credentials.dart';
import 'package:http/http.dart' as http;

class P2PVideo extends StatefulWidget {
  const P2PVideo({Key? key}) : super(key: key);

  @override
  _P2PVideoState createState() => _P2PVideoState();
}

bool _mirror = false;

class _P2PVideoState extends State<P2PVideo> {
  RTCPeerConnection? _peerConnection;
  final _localRenderer = RTCVideoRenderer();

  MediaStream? _localStream;

  RTCDataChannelInit? _dataChannelDict;
  RTCDataChannel? _dataChannel;
  String transformType = "none";

  // MediaStream? _localStream;
  bool _inCalling = false;

  DateTime? _timeStart;

  bool _loading = false;

  void _onTrack(RTCTrackEvent event) {
    print("TRACK EVENT: ${event.streams.map((e) => e.id)}, ${event.track.id}");
    if (event.track.kind == "video") {
      print("HERE");
      _localRenderer.srcObject = event.streams[0];
    }
  }

  void _onDataChannelState(RTCDataChannelState? state) {
    switch (state) {
      case RTCDataChannelState.RTCDataChannelClosed:
        print("Camera Closed!!!!!!!");
        break;
      case RTCDataChannelState.RTCDataChannelOpen:
        print("Camera Opened!!!!!!!");
        break;
      default:
        print("Data Channel State: $state");
    }
  }

  Future<bool> _waitForGatheringComplete(_) async {
    print("WAITING FOR GATHERING COMPLETE");
    if (_peerConnection!.iceGatheringState ==
        RTCIceGatheringState.RTCIceGatheringStateComplete) {
      return true;
    } else {
      await Future.delayed(Duration(seconds: 1));
      return await _waitForGatheringComplete(_);
    }
  }

  void _toggleCamera() async {
    if (_localStream == null) throw Exception('Stream is not initialized');

    /** Instruct server to return mirrored/unmirrored image */
    _mirror = !_mirror;
    _setMirror(_mirror);
    /** */
    
    final videoTrack = _localStream!
        .getVideoTracks()
        .firstWhere((track) => track.kind == 'video');
    await Helper.switchCamera(videoTrack);
  }

  Future<void> _negotiateRemoteConnection() async {
    return _peerConnection!
        .createOffer()
        .then((offer) {
          return _peerConnection!.setLocalDescription(offer);
        })
        .then(_waitForGatheringComplete)
        .then((_) async {
          var des = await _peerConnection!.getLocalDescription();
          var headers = {
            'Content-Type': 'application/json',
          };
          var request = http.Request(
            'POST',
            Uri.parse(
                'http://192.168.1.252:8080/offer'), // CHANGE URL HERE TO LOCAL SERVER
          );
          request.body = json.encode(
            {
              "sdp": des!.sdp,
              "type": des.type,
              "mirror": "false",
              "video_transform": transformType,
            },
          );
          request.headers.addAll(headers);

          http.StreamedResponse response = await request.send();

          String data = "";
          print(response);
          if (response.statusCode == 200) {
            data = await response.stream.bytesToString();
            var dataMap = json.decode(data);
            print(dataMap);
            await _peerConnection!.setRemoteDescription(
              RTCSessionDescription(
                dataMap["sdp"],
                dataMap["type"],
              ),
            );
          } else {
            print(response.reasonPhrase);
          }
        });
  }

  Future<void> _makeCall() async {
    setState(() {
      _loading = true;
    });
    /** Instruct server to return mirrored/unmirrored image */
    _mirror = false;
    _setMirror(_mirror);
    /** */
    var configuration = <String, dynamic>{
      'sdpSemantics': 'unified-plan',
      // 'iceServers': [
      //   {
      //     'url': 'turn:16.163.180.160:3478',
      //     'username': USER,
      //     'credential': CREDENTIAL
      //   },
      // ]
    };

    //* Create Peer Connection
    if (_peerConnection != null) return;
    _peerConnection = await createPeerConnection(
      configuration,
    );

    _peerConnection!.onTrack = _onTrack;
    // _peerConnection!.onAddTrack = _onAddTrack;

    //* Create Data Channel
    _dataChannelDict = RTCDataChannelInit();
    _dataChannelDict!.ordered = true;
    _dataChannel = await _peerConnection!.createDataChannel(
      "chat",
      _dataChannelDict!,
    );
    _dataChannel!.onDataChannelState = _onDataChannelState;
    // _dataChannel!.onMessage = _onDataChannelMessage;

    final mediaConstraints = <String, dynamic>{
      'audio': false,
      'video': {
        'mandatory': {
          'minWidth':
              '480', // Provide your own width, height and frame rate here
          'minHeight': '720',
          'minFrameRate': '30',
        },
        // 'facingMode': 'user',
        'facingMode': 'environment',
        'optional': [],
      }
    };

    try {
      var stream = await navigator.mediaDevices.getUserMedia(mediaConstraints);
      // _mediaDevicesList = await navigator.mediaDevices.enumerateDevices();
      _localStream = stream;
      // _localRenderer.srcObject = _localStream;

      stream.getTracks().forEach((element) {
        _peerConnection!.addTrack(element, stream);
      });

      print("NEGOTIATE");
      await _negotiateRemoteConnection();
    } catch (e) {
      print(e.toString());
    }
    if (!mounted) return;

    setState(() {
      _inCalling = true;
      _loading = false;
    });
  }

  Future<void> _stopCall() async {
    try {
      // await _localStream?.dispose();
      await _dataChannel?.close();
      await _peerConnection?.close();
      _peerConnection = null;
      _localRenderer.srcObject = null;
    } catch (e) {
      print(e.toString());
    }
    setState(() {
      _inCalling = false;
    });
  }

  Future<void> initLocalRenderers() async {
    await _localRenderer.initialize();
  }

  void _setMirror(bool mirror) async {
    var headers = {
      'Content-Type': 'application/json',
    };
    var request = http.Request(
      'POST',
      Uri.parse(
          'http://192.168.1.252:8080/set-mirror'), // CHANGE URL HERE TO LOCAL SERVER
    );
    request.body = json.encode(
      {
        "mirror": mirror.toString(),
      },
    );
    request.headers.addAll(headers);

    http.StreamedResponse response = await request.send();
    String data = "";
    print(response);
    if (response.statusCode == 200) {
      data = await response.stream.bytesToString();
      var dataMap = json.decode(data);
      print(dataMap);
    }
  }

  void _setTask(String task) async {
    var headers = {
      'Content-Type': 'application/json',
    };
    var request = http.Request(
      'POST',
      Uri.parse(
          'http://192.168.1.252:8080/set-task'), // CHANGE URL HERE TO LOCAL SERVER
    );
    request.body = json.encode(
      {
        "task": task,
      },
    );
    request.headers.addAll(headers);

    http.StreamedResponse response = await request.send();
    String data = "";
    print(response);
    if (response.statusCode == 200) {
      data = await response.stream.bytesToString();
      var dataMap = json.decode(data);
      print(dataMap);
    }
  }

  @override
  void initState() {
    super.initState();

    initLocalRenderers();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: OrientationBuilder(
        builder: (context, orientation) {
          return SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(10),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Padding(
                    padding: const EdgeInsets.all(5),
                    child: ConstrainedBox(
                      // height: MediaQuery.of(context).size.width > 500
                      //     ? 500
                      //     : MediaQuery.of(context).size.width - 20,
                      constraints: BoxConstraints(maxHeight: 500),
                      // width: MediaQuery.of(context).size.width > 500
                      //     ? 500
                      //     : MediaQuery.of(context).size.width - 20,
                      child: AspectRatio(
                        aspectRatio: 3 / 4,
                        child: Stack(
                          children: [
                            Positioned.fill(
                              child: Container(
                                color: Colors.black,
                                child: _loading
                                    ? Center(
                                        child: CircularProgressIndicator(
                                          strokeWidth: 4,
                                        ),
                                      )
                                    : Container(),
                              ),
                            ),
                            Positioned.fill(
                              child: RTCVideoView(
                                _localRenderer,
                                // mirror: true,
                              ),
                            ),
                            _inCalling
                                ? Align(
                                    alignment: Alignment.bottomRight,
                                    child: InkWell(
                                      onTap: _toggleCamera,
                                      child: Container(
                                        height: 50,
                                        width: 50,
                                        color: Colors.black26,
                                        child: Center(
                                          child: Icon(
                                            Icons.cameraswitch,
                                            color: Colors.grey,
                                          ),
                                        ),
                                      ),
                                    ),
                                  )
                                : Container(),
                          ],
                        ),
                      ),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Wrap(
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text("Transformation: "),
                            DropdownButton(
                              value: transformType,
                              onChanged: (value) {
                                setState(() {
                                  transformType = value.toString();
                                });
                                _setTask(value.toString());
                              },
                              items: ["none", "edges", "cartoon", "rotate", "object detection"]
                                  .map(
                                    (e) => DropdownMenuItem(
                                      value: e,
                                      child: Text(
                                        e,
                                      ),
                                    ),
                                  )
                                  .toList(),
                            ),
                          ],
                        ),
                        SizedBox(
                          width: 20,
                        ),
                      ],
                    ),
                  ),
                  Expanded(child: Container()),
                  InkWell(
                    onTap: _loading
                        ? () {}
                        : _inCalling
                            ? _stopCall
                            : _makeCall,
                    child: Container(
                      decoration: BoxDecoration(
                        color: _loading
                            ? Colors.amber
                            : _inCalling
                                ? Colors.red
                                : Theme.of(context).primaryColor,
                        borderRadius: BorderRadius.circular(15),
                      ),
                      child: Padding(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 5),
                        child: _loading
                            ? Padding(
                                padding: const EdgeInsets.all(8.0),
                                child: CircularProgressIndicator(),
                              )
                            : Text(
                                _inCalling ? "STOP" : "START",
                                style: TextStyle(
                                  fontSize: 24,
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }
}
