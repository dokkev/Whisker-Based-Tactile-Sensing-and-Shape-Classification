#ifndef PHYSICS_SERVER_COMMAND_PROCESSOR_H
#define PHYSICS_SERVER_COMMAND_PROCESSOR_H

#include "LinearMath/btVector3.h"

#include "PhysicsCommandProcessorInterface.h"

struct SharedMemLines
{
	btVector3 m_from;
	btVector3 m_to;
	btVector3 m_color;
};



///todo: naming. Perhaps PhysicsSdkCommandprocessor?
class PhysicsServerCommandProcessor : public CommandProcessorInterface
{

	struct PhysicsServerCommandProcessorInternalData* m_data;

	void resetSimulation();

protected:


	bool processStateLoggingCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRequestCameraImageCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processSaveWorldCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processCreateCollisionShapeCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processCreateVisualShapeCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processCustomCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processUserDebugDrawCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processSetVRCameraStateCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRequestVREventsCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRequestMouseEventsCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRequestKeyboardEventsCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRequestRaycastIntersectionsCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRequestDebugLinesCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processSyncBodyInfoCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processSendDesiredStateCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRequestActualStateCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRequestContactpointInformationCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRequestBodyInfoCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processLoadSDFCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processCreateMultiBodyCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processLoadURDFCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processLoadBunnyCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processCreateSensorCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processProfileTimingCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRequestCollisionInfoCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processForwardDynamicsCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRequestInternalDataCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processChangeDynamicsInfoCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processSetAdditionalSearchPathCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processGetDynamicsInfoCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRequestPhysicsSimulationParametersCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processSendPhysicsParametersCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processInitPoseCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processResetSimulationCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processCreateRigidBodyCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processPickBodyCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processMovePickedBodyCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRemovePickingConstraintCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRequestAabbOverlapCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRequestOpenGLVisualizeCameraCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processConfigureOpenGLVisualizerCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processInverseDynamicsCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processCalculateJacobianCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processCalculateMassMatrixCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processApplyExternalForceCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRemoveBodyCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processCreateUserConstraintCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processCalculateInverseKinematicsCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processRequestVisualShapeInfoCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processUpdateVisualShapeCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processChangeTextureCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processLoadTextureCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processLoadBulletCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processSaveBulletCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);
	bool processLoadMJCFCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);


	bool loadSdf(const char* fileName, char* bufferServerToClient, int bufferSizeInBytes, bool useMultiBody, int flags, btScalar globalScaling);

	bool loadUrdf(const char* fileName, const class btVector3& pos, const class btQuaternion& orn,
		bool useMultiBody, bool useFixedBase, int* bodyUniqueIdPtr, char* bufferServerToClient, int bufferSizeInBytes, int flags, btScalar globalScaling);

	bool loadMjcf(const char* fileName, char* bufferServerToClient, int bufferSizeInBytes, bool useMultiBody, int flags);

	bool processImportedObjects(const char* fileName, char* bufferServerToClient, int bufferSizeInBytes, bool useMultiBody, int flags, class URDFImporterInterface& u2b);

	bool	supportsJointMotor(class btMultiBody* body, int linkIndex);

	int createBodyInfoStream(int bodyUniqueId, char* bufferServerToClient, int bufferSizeInBytes);
	void deleteCachedInverseDynamicsBodies();
	void deleteCachedInverseKinematicsBodies();
	void deleteStateLoggers();

public:
	PhysicsServerCommandProcessor();
	virtual ~PhysicsServerCommandProcessor();

	void	createJointMotors(class btMultiBody* body);

	virtual void createEmptyDynamicsWorld();
	virtual void deleteDynamicsWorld();

	virtual bool connect()
	{
		return true;
	};

	virtual void disconnect() {}

	virtual bool isConnected() const
	{
		return true;
	}



	virtual bool processCommand(const struct SharedMemoryCommand& clientCmd, struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes);

	virtual bool receiveStatus(struct SharedMemoryStatus& serverStatusOut, char* bufferServerToClient, int bufferSizeInBytes)
	{
		return false;
	};

	virtual void renderScene(int renderFlags);
	virtual void   physicsDebugDraw(int debugDrawFlags);
	virtual void setGuiHelper(struct GUIHelperInterface* guiHelper);
	virtual void syncPhysicsToGraphics();


	//@todo(erwincoumans) Should we have shared memory commands for picking objects?
	///The pickBody method will try to pick the first body along a ray, return true if succeeds, false otherwise
	virtual bool pickBody(const btVector3& rayFromWorld, const btVector3& rayToWorld);
	virtual bool movePickedBody(const btVector3& rayFromWorld, const btVector3& rayToWorld);
	virtual void removePickingConstraint();

	//logging /playback the shared memory commands
	virtual void enableCommandLogging(bool enable, const char* fileName);
	virtual void replayFromLogFile(const char* fileName);
	virtual void replayLogCommand(char* bufferServerToClient, int bufferSizeInBytes );

	//logging of object states (position etc)
	void tickPlugins(btScalar timeStep, bool isPreTick);
	void logObjectStates(btScalar timeStep);
	void processCollisionForces(btScalar timeStep);

	virtual void stepSimulationRealTime(double dtInSec,const struct b3VRControllerEvent* vrControllerEvents, int numVRControllerEvents, const struct b3KeyboardEvent* keyEvents, int numKeyEvents, const struct b3MouseEvent* mouseEvents, int numMouseEvents);

	virtual void enableRealTimeSimulation(bool enableRealTimeSim);
	virtual bool isRealTimeSimulationEnabled() const;

	void applyJointDamping(int bodyUniqueId);

	virtual void setTimeOut(double timeOutInSeconds);

	virtual const btVector3& getVRTeleportPosition() const;
	virtual void setVRTeleportPosition(const btVector3& vrTeleportPos);

	virtual const btQuaternion& getVRTeleportOrientation() const;
	virtual void setVRTeleportOrientation(const btQuaternion& vrTeleportOrn);
};

#endif //PHYSICS_SERVER_COMMAND_PROCESSOR_H
